# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OCR geometry over the installed MiniCPM-V 4.6 video pipeline."""

import copy

from transformers.feature_extraction_utils import BatchFeature
from transformers.models.minicpmv4_6.video_processing_minicpmv4_6 import (
    MiniCPMV4_6VideoProcessor,
)
from transformers.models.minicpmv4_6.video_processing_minicpmv4_6 import (
    MiniCPMV4_6VideoProcessorKwargs as MiniCPMOcrVideoProcessorKwargs,
)
from transformers.processing_utils import Unpack
from transformers.video_utils import VideoInput

from .geometry import MiniCPMOcrGeometry


class MiniCPMOcrVideoProcessor(MiniCPMOcrGeometry, MiniCPMV4_6VideoProcessor):
    downsample_mode = "4x"

    def __call__(
        self,
        videos: VideoInput,
        **kwargs: Unpack[MiniCPMOcrVideoProcessorKwargs],
    ) -> BatchFeature:
        # `downsample_mode` also selects the patch grid `find_best_resize` aligns to, and that grid
        # is read from the instance. Run the override on a shallow copy so the processor itself is
        # never mutated and concurrent calls cannot observe each other's mode.
        downsample_mode = kwargs.pop("downsample_mode", None)
        processor = self
        if downsample_mode is not None and downsample_mode != self.downsample_mode:
            processor = copy.copy(self)
            processor.downsample_mode = downsample_mode
        return processor.preprocess(videos, **kwargs)


__all__ = ["MiniCPMOcrVideoProcessor"]
