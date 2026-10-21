# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OCR grid geometry shared by its image and video processors."""

import math

from transformers.models.minicpmv4_6.image_processing_minicpmv4_6 import ensure_divide


class MiniCPMOcrGeometry:
    """Use one 2x2 merge for 4x and preserve OCR row/column tie breaking."""

    def find_best_resize(
        self,
        image_size: tuple[int, int],
        scale_resolution: int,
        patch_size: int,
        allow_upscale: bool = False,
    ) -> tuple[int, int]:
        height, width = image_size
        if (height * width > scale_resolution * scale_resolution) or allow_upscale:
            aspect_ratio = width / height
            height = int(scale_resolution / math.sqrt(aspect_ratio))
            width = int(height * aspect_ratio)
        # The resize grid follows the number of 2×2 spatial merges the visual tokens go through:
        # one in `"4x"` mode (downsample MLP), two in `"16x"` mode (ViT insert merger as well).
        merge_factor = 2 if self.downsample_mode == "4x" else 4
        best_width = ensure_divide(width, patch_size * merge_factor)
        best_height = ensure_divide(height, patch_size * merge_factor)
        return best_height, best_width

    def get_sliced_grid(
        self,
        image_size: tuple[int, int],
        max_slice_nums: int,
        scale_resolution: int,
    ) -> list[int] | None:
        original_height, original_width = image_size
        log_ratio = math.log(original_width / original_height)
        ratio = original_width * original_height / (scale_resolution * scale_resolution)
        multiple = min(math.ceil(ratio), max_slice_nums)
        if multiple <= 1:
            return None

        best_grid = [1, 1]
        min_error = float("inf")
        for num_slices in [multiple - 1, multiple, multiple + 1]:
            if num_slices == 1 or num_slices > max_slice_nums:
                continue
            for num_rows in range(1, num_slices + 1):
                if num_slices % num_rows == 0:
                    num_cols = num_slices // num_rows
                    error = abs(log_ratio - math.log(num_cols / num_rows))
                    if error < min_error:
                        best_grid = [num_rows, num_cols]
                        min_error = error
                    elif error == min_error and num_rows > best_grid[0]:
                        best_grid = [num_rows, num_cols]
        return best_grid
