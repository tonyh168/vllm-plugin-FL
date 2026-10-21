# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OCR-specific geometry and batching over vLLM's MiniCPM-V 4.6 stack.

The language/vision towers, packing and video formatting remain upstream.
OCR adds pixel-dependent image placeholders and per-item 4x/16x batching.
"""

import copy
from dataclasses import replace
from functools import partial

import torch
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.minicpmv import (
    MiniCPMVDummyInputsBuilder,
    MiniCPMVImageEmbeddingItems,
    MiniCPMVImagePixelInputs,
)
from vllm.model_executor.models.minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
    MiniCPMV4_6MultiModalProcessor,
    MiniCPMV4_6ProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import ImageProcessorItems, ImageSize
from vllm.multimodal.processing.processor import (
    PromptUpdateDetails,
)
from vllm.multimodal.processing.processor import _cached_encode as cached_encode

_SCOPES = ("images_kwargs", "videos_kwargs", "audio_kwargs")
_IMAGE_UPSIZE_KWARGS = (
    "disable_small_image_upsize",
    "adaptive_small_image_upsize",
    "adaptive_upsize_font_px",
    "adaptive_upsize_dense_per_kpx",
    "adaptive_upsize_tiny_pixels",
    "adaptive_upsize_tiny_target_resolution",
)


def _expand_flags(value: object, num_items: int) -> list[bool | None]:
    """Normalize one scalar or one flag per item without collapsing mixed modes."""
    if value is None:
        return [None] * num_items
    values = value if isinstance(value, (list, tuple)) else [value]
    flags = []
    for item in values:
        if isinstance(item, torch.Tensor):
            flags.extend(item.reshape(-1).tolist())
        else:
            flags.append(item)
    if len(flags) == 1:
        flags *= num_items
    if len(flags) != num_items:
        raise ValueError("vision merger flag count does not match image count")
    return [bool(flag) for flag in flags]


def _select_vision_items(image_input, indices, num_slices):
    offsets = [0]
    for count in num_slices:
        offsets.append(offsets[-1] + count)
    pixels, sizes = [], []
    for i in indices:
        start, end = offsets[i : i + 2]
        pixels.extend(image_input["pixel_values"][start:end])
        sizes.append(image_input["tgt_sizes"][start:end])
    return MiniCPMVImagePixelInputs(
        type="pixel_values",
        pixel_values=pixels,
        tgt_sizes=torch.cat(sizes, dim=0),
        num_slices=torch.tensor([num_slices[i] for i in indices]),
    )


class MiniCPMOcrProcessingInfo(MiniCPMV4_6ProcessingInfo):
    def _pinned_image_processor(self, downsample_mode=None):
        processor = self.get_image_processor()
        mode = self._get_downsample_mode(downsample_mode)
        if processor.downsample_mode != mode:
            processor = copy.copy(processor)
            processor.downsample_mode = mode
        return processor

    def get_image_size_with_most_features(self):
        resolution = self.get_image_processor().scale_resolution
        return ImageSize(resolution, resolution * self.get_image_max_slice_num())

    def get_video_frame_size_with_most_features(self):
        resolution = self.get_image_processor().scale_resolution
        return ImageSize(resolution, resolution * self.get_video_max_slice_num())

    def _compute_visual_tokens(
        self, image_size, max_slice_nums=None, downsample_mode=None
    ):
        processor = self._pinned_image_processor(downsample_mode)
        max_slices = (
            processor.max_slice_nums if max_slice_nums is None else max_slice_nums
        )
        size = (image_size.height, image_size.width)
        resolution, patch = processor.scale_resolution, processor.patch_size
        divisor = patch * patch * (4 if processor.downsample_mode == "4x" else 16)
        grid = processor.get_sliced_grid(size, max_slices, resolution)
        height, width = processor.find_best_resize(
            size, resolution, patch, allow_upscale=grid is None
        )
        source_tokens = height * width // divisor
        if grid is None:
            return [0, 0], source_tokens, 0
        height, width = processor.get_refine_size(
            size, grid, resolution, patch, allow_upscale=True
        )
        rows, cols = grid
        patch_tokens = (height // rows) * (width // cols) // divisor
        # HF geometry is [rows, cols]; vLLM's placeholder contract is [cols, rows].
        return [cols, rows], source_tokens, patch_tokens

    def _merge_and_resolve_mm_processor_kwargs(self, mm_kwargs):
        merged = self.ctx.get_merged_mm_kwargs(mm_kwargs)
        flat = {key: value for key, value in merged.items() if key not in _SCOPES}
        return {
            scope: {**flat, **merged.get(scope, {})}
            for scope in ("images_kwargs", "videos_kwargs")
        }


class MiniCPMOcrMultiModalProcessor(MiniCPMV4_6MultiModalProcessor):
    def _processor_kwargs(self, mm_kwargs, modality):
        scope = "images_kwargs" if modality == "image" else "videos_kwargs"
        kwargs = dict(
            self.info._merge_and_resolve_mm_processor_kwargs(mm_kwargs)[scope]
        )
        mode = kwargs.get("downsample_mode")
        kwargs["downsample_mode"] = (
            self.info._get_downsample_mode() if mode is None else str(mode)
        )
        if modality == "video":
            for key in _IMAGE_UPSIZE_KWARGS:
                kwargs.pop(key, None)
        return kwargs

    def process_images(self, mm_data, mm_kwargs, tok_kwargs):
        return super().process_images(
            mm_data, self._processor_kwargs(mm_kwargs, "image"), tok_kwargs
        )

    def process_videos(self, mm_data, mm_kwargs, tok_kwargs):
        kwargs = self._processor_kwargs(mm_kwargs, "video")
        inputs = super().process_videos(mm_data, kwargs, tok_kwargs)
        if inputs:
            flag = (
                kwargs["downsample_mode"] != "4x"
                and getattr(self.info.get_hf_config(), "insert_layer_id", -1) >= 0
            )
            inputs["video_use_vit_merger"] = [
                torch.tensor([flag], dtype=torch.bool)
                for _ in inputs["video_pixel_values"]
            ]
        return inputs

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        fields = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        for name, modality in (
            ("use_vit_merger", "image"),
            ("video_use_vit_merger", "video"),
            ("video_image_sizes", "video"),
        ):
            if name in hf_inputs:
                fields[name] = MultiModalFieldConfig.batched(modality, keep_on_cpu=True)
        return fields

    def _image_prompt_text(self, sizes, image_size, image_idx, kwargs):
        """Use actual patch grids: OCR's adaptive resize depends on image pixels."""
        if sizes.ndim != 2 or sizes.shape[-1] != 2 or sizes.shape[0] < 1:
            return None
        mode = kwargs["downsample_mode"]
        grid, _, _ = self.info._compute_visual_tokens(
            image_size, kwargs.get("max_slice_nums"), downsample_mode=mode
        )
        cols, rows = grid
        if rows * cols + 1 != sizes.shape[0]:
            return None
        counts = (sizes.to(torch.long).prod(-1) // (4 if mode == "4x" else 16)).tolist()
        if counts[0] <= 0:
            return None
        tokenizer = self.info.get_tokenizer()
        text = (
            f"{tokenizer.image_id_start_token}{image_idx}{tokenizer.image_id_end_token}"
            + tokenizer.image_start_token
            + tokenizer.image_token * counts[0]
            + tokenizer.image_end_token
        )
        if rows and cols and counts[1] > 0:
            patch = (
                tokenizer.slice_start_token
                + tokenizer.image_token * counts[1]
                + tokenizer.slice_end_token
            )
            text += "\n".join([patch * cols for _ in range(rows)])
        return text

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        tokenizer = self.info.get_tokenizer()
        image_kwargs = self._processor_kwargs(hf_processor_mm_kwargs, "image")
        video_kwargs = self._processor_kwargs(hf_processor_mm_kwargs, "video")

        def image_replacement(index):
            images = mm_items.get_items(
                "image", (MiniCPMVImageEmbeddingItems, ImageProcessorItems)
            )
            size = images.get_image_size(index)
            items = out_mm_kwargs.get("image")
            field = (
                items[index].get("tgt_sizes")
                if items is not None and index < len(items)
                else None
            )
            text = None
            if field is not None and isinstance(field.data, torch.Tensor):
                text = self._image_prompt_text(field.data, size, index, image_kwargs)
            if text is None:
                text = self.info.get_slice_image_placeholder(
                    size,
                    image_idx=index,
                    max_slice_nums=image_kwargs.get("max_slice_nums"),
                    downsample_mode=image_kwargs["downsample_mode"],
                )
            return PromptUpdateDetails.from_seq(text)

        def tokenized_replacement(index, replacement, embed_token):
            details = replacement(index)
            return PromptUpdateDetails.select_token_id(
                cached_encode(tokenizer, details.full, add_special_tokens=False),
                tokenizer.get_vocab()[embed_token],
            )

        # The parent owns placeholder discovery and processed-video frame sizes.
        # Its image callback is replaced because OCR counts depend on actual pixels.
        updates = super()._get_prompt_updates(mm_items, video_kwargs, out_mm_kwargs)
        return [
            replace(
                update,
                target=cached_encode(
                    tokenizer, update.target, add_special_tokens=False
                ),
                replacement=partial(
                    tokenized_replacement,
                    replacement=(
                        image_replacement
                        if update.modality == "image"
                        else update.replacement
                    ),
                    embed_token=(
                        tokenizer.image_token
                        if update.modality == "image"
                        else tokenizer.video_token
                    ),
                ),
            )
            for update in updates
        ]


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMOcrMultiModalProcessor,
    info=MiniCPMOcrProcessingInfo,
    dummy_inputs=MiniCPMVDummyInputsBuilder,
)
class MiniCPMOcrForConditionalGeneration(MiniCPMV4_6ForConditionalGeneration):
    def _process_vision_input(self, image_input, use_vit_merger=None):
        if image_input["type"] == "image_embeds":
            return image_input["image_embeds"]
        counts = image_input["num_slices"].tolist()
        flags = _expand_flags(use_vit_merger, len(counts))
        if len(set(flags)) == 1:
            return super()._process_vision_input(image_input, flags[0])
        encoded = {}
        for flag in dict.fromkeys(flags):
            indices = [i for i, item_flag in enumerate(flags) if item_flag == flag]
            inputs = _select_vision_items(image_input, indices, counts)
            features = super()._process_vision_input(inputs, flag)
            encoded.update(zip(indices, features))
        return [encoded[i] for i in range(len(counts))]

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        embeddings = []
        for prefix in ("", "video_"):
            inputs = {
                key: kwargs[prefix + key]
                for key in ("pixel_values", "image_embeds", "tgt_sizes")
                if prefix + key in kwargs
            }
            if (
                inputs.get("pixel_values") is None
                and inputs.get("image_embeds") is None
            ):
                continue
            image_input = self._parse_and_validate_vision_input(**inputs)
            if image_input is not None:
                embeddings.extend(
                    self._process_vision_input(
                        image_input, kwargs.get(prefix + "use_vit_merger")
                    )
                )
        return tuple(embeddings) if embeddings else []

    def load_weights(self, weights):
        # The upstream child loader returns unfused names; report the actual
        # initialized parameter so vLLM's missing-weight validation stays exact.
        loaded_qkv = set()
        params = dict(self.named_parameters())
        mapper = self.hf_to_vllm_mapper

        def remaining():
            for name, tensor in weights:
                mapped = mapper._map_name(name)
                for projection, shard in (
                    ("q_proj", "q"),
                    ("k_proj", "k"),
                    ("v_proj", "v"),
                ):
                    prefix = "vit_merger.self_attn." + projection + "."
                    if mapped is not None and mapped.startswith(prefix):
                        target = mapped.replace(projection, "qkv_proj", 1)
                        param = params[target]
                        param.weight_loader(param, tensor, shard)
                        loaded_qkv.add(target)
                        break
                else:
                    yield name, tensor

        return super().load_weights(remaining()) | loaded_qkv
