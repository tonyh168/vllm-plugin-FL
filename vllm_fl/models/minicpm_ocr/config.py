# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM-OCR compatibility config for Transformers before native OCR support."""

from transformers import MiniCPMV4_6Config


class MiniCPMOcrConfig(MiniCPMV4_6Config):
    model_type = "minicpm_ocr"

    def __init__(self, **kwargs):
        kwargs.setdefault("downsample_mode", "4x")
        kwargs.setdefault("drop_vision_last_layer", False)
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(**kwargs)
