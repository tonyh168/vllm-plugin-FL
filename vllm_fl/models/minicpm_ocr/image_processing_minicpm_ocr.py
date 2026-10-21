# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OCR-only resize options over the installed MiniCPM-V 4.6 image processor."""

import copy

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.models.minicpmv4_6.image_processing_minicpmv4_6 import (
    MiniCPMV4_6ImageProcessor,
    MiniCPMV4_6ImageProcessorKwargs,
)
from transformers.processing_utils import Unpack
from transformers.utils import requires_backends

from .geometry import MiniCPMOcrGeometry


class MiniCPMOcrImageProcessorKwargs(MiniCPMV4_6ImageProcessorKwargs, total=False):
    disable_small_image_upsize: bool
    adaptive_small_image_upsize: bool
    adaptive_upsize_font_px: float
    adaptive_upsize_dense_per_kpx: float
    adaptive_upsize_tiny_pixels: int
    adaptive_upsize_tiny_target_resolution: int


def adaptive_upsize_tier(
    image: np.ndarray,
    font_px_threshold: float,
    dense_per_kpx_threshold: float,
    tiny_pixels: int,
) -> str:
    r"""
    Pick the resize target for an image small enough that it is not sliced.

    Text that is both small and dense is worth upscaling to the full `scale_resolution`
    (`"448"`), images that are merely tiny are upscaled to a lower target (`"84"`), and
    everything else keeps its native resolution (`"native"`). Glyph height and density are
    measured with an Otsu threshold followed by 8-connected component analysis, matching the
    classifier the checkpoint was trained with, so OpenCV is required.

    Args:
        image (`np.ndarray`):
            `uint8` image to classify, shaped `(num_channels, height, width)` or `(height, width)`.
        font_px_threshold (`float`):
            Largest median glyph height, in pixels, still considered small text.
        dense_per_kpx_threshold (`float`):
            Smallest number of glyphs per 1000 pixels still considered dense text.
        tiny_pixels (`int`):
            Images with fewer pixels than this are treated as tiny.
    """
    requires_backends(adaptive_upsize_tier, ["cv2"])
    import cv2

    height, width = image.shape[-2:]
    if height <= 0 or width <= 0:
        return "native"

    if image.ndim == 3 and image.shape[0] >= 3:
        channels = image.astype(np.int32)
        # ITU-R 601-2 luma, matching the integer arithmetic of `PIL.Image.convert("L")`
        gray = (
            (channels[0] * 19595 + channels[1] * 38470 + channels[2] * 7471 + 32768)
            >> 16
        ).astype(np.uint8)
    else:
        gray = np.ascontiguousarray(image.reshape(height, width)).astype(np.uint8)

    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground = binarized == 0
    if foreground.mean() > 0.5:
        # Assume glyphs cover less than half the image, so flip light text on a dark page
        foreground = ~foreground

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8
    )
    glyph_heights = []
    for label in range(1, num_labels):
        _, _, component_width, component_height, area = stats[label]
        if area < 3 or component_height < 2:
            continue
        if component_width >= 0.9 * width or component_height >= 0.9 * height:
            continue
        glyph_heights.append(component_height)

    font_px = float(np.median(glyph_heights)) if glyph_heights else 0.0
    density = len(glyph_heights) * 1000.0 / (width * height)
    if 0 < font_px <= font_px_threshold and density >= dense_per_kpx_threshold:
        return "448"

    if height * width < tiny_pixels:
        return "84"
    return "native"


class MiniCPMOcrImageProcessor(MiniCPMOcrGeometry, MiniCPMV4_6ImageProcessor):
    downsample_mode = "4x"
    valid_kwargs = MiniCPMOcrImageProcessorKwargs
    disable_small_image_upsize = False
    adaptive_small_image_upsize = False
    adaptive_upsize_font_px = 12.0
    adaptive_upsize_dense_per_kpx = 2.0
    adaptive_upsize_tiny_pixels = 6272
    adaptive_upsize_tiny_target_resolution = 84

    def find_best_resize(
        self, image_size, scale_resolution, patch_size, allow_upscale=False
    ):
        # Only a call-local, unsliced image carries this source resize choice.
        # The shared processor and sliced-image refinement never receive it.
        source_resize = getattr(self, "_source_resize", None)
        if source_resize is not None:
            scale_resolution, allow_upscale = source_resize
        return super().find_best_resize(
            image_size, scale_resolution, patch_size, allow_upscale=allow_upscale
        )

    def _preprocess(
        self,
        images,
        scale_resolution,
        max_slice_nums,
        slice_mode,
        return_tensors=None,
        **kwargs,
    ) -> BatchFeature:
        kwargs.update(
            scale_resolution=scale_resolution,
            max_slice_nums=max_slice_nums,
            slice_mode=slice_mode,
        )
        if not (self.disable_small_image_upsize or self.adaptive_small_image_upsize):
            return super()._preprocess(images, return_tensors=return_tensors, **kwargs)

        batches = []
        for image in images:
            grid = (
                self.get_sliced_grid(image.shape[-2:], max_slice_nums, scale_resolution)
                if slice_mode
                else None
            )
            processor = self
            if grid is None:
                processor = copy.copy(self)
                processor._source_resize = self.resolve_source_resolution(
                    image, scale_resolution
                )
            # Reuse upstream slicing, normalization and patch packing. Keep
            # tensors until the complete batch is assembled, then convert once.
            batches.append(
                super(MiniCPMOcrImageProcessor, processor)._preprocess(
                    [image], return_tensors=None, **kwargs
                )
            )

        return BatchFeature(
            data={
                "pixel_values": torch.cat([b["pixel_values"] for b in batches], dim=-1),
                "target_sizes": torch.cat([b["target_sizes"] for b in batches], dim=0),
                **{
                    key: [item for b in batches for item in b[key]]
                    for key in ("grids", "num_patches_per_image")
                },
            },
            tensor_type=return_tensors,
            skip_tensor_conversion=["grids", "num_patches_per_image"],
        )

    def __call__(
        self,
        images: ImageInput,
        **kwargs: Unpack[MiniCPMOcrImageProcessorKwargs],
    ) -> BatchFeature:
        # These options are read from the instance by `find_best_resize` and
        # `resolve_source_resolution` rather than received as `_preprocess` arguments. Apply the
        # overrides to a shallow copy so the processor itself is never mutated and concurrent calls
        # cannot observe each other's values.
        processor = self
        for name in (
            "downsample_mode",
            "disable_small_image_upsize",
            "adaptive_small_image_upsize",
            "adaptive_upsize_font_px",
            "adaptive_upsize_dense_per_kpx",
            "adaptive_upsize_tiny_pixels",
            "adaptive_upsize_tiny_target_resolution",
        ):
            value = kwargs.pop(name, None)
            if value is None or value == getattr(self, name):
                continue
            if processor is self:
                processor = copy.copy(self)
            setattr(processor, name, value)
        return processor.preprocess(images, **kwargs)

    def resolve_source_resolution(
        self, image: "torch.Tensor", scale_resolution: int
    ) -> tuple[int, bool]:
        r"""
        Resolve the resize target of an image that is small enough not to be sliced.

        Returns the resolution to resize towards and whether upscaling is allowed. Text that is
        small and dense is upscaled to `scale_resolution`, tiny images are upscaled to
        `adaptive_upsize_tiny_target_resolution`, and the remaining ones keep their native resolution.

        Args:
            image (`torch.Tensor`):
                Image to classify, shaped `(num_channels, height, width)`.
            scale_resolution (`int`):
                Target resolution requested for this call.
        """
        if not self.adaptive_small_image_upsize:
            return scale_resolution, not self.disable_small_image_upsize

        tier = adaptive_upsize_tier(
            image.detach().to("cpu", torch.uint8).numpy(),
            font_px_threshold=self.adaptive_upsize_font_px,
            dense_per_kpx_threshold=self.adaptive_upsize_dense_per_kpx,
            tiny_pixels=self.adaptive_upsize_tiny_pixels,
        )
        if tier == "448":
            return scale_resolution, True
        if tier == "84":
            return self.adaptive_upsize_tiny_target_resolution, True
        return scale_resolution, not self.disable_small_image_upsize


__all__ = ["MiniCPMOcrImageProcessor"]
