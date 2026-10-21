# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OCR processors for the Transformers version shipped with vLLM 0.24."""

from transformers import AutoTokenizer
from transformers.models.minicpmv4_6.processing_minicpmv4_6 import MiniCPMV4_6Processor

from .image_processing_minicpm_ocr import MiniCPMOcrImageProcessor
from .video_processing_minicpm_ocr import MiniCPMOcrVideoProcessor


class MiniCPMOcrProcessor(MiniCPMV4_6Processor):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from pathlib import Path

        image = MiniCPMOcrImageProcessor.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        video = MiniCPMOcrVideoProcessor.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        template = Path(pretrained_model_name_or_path) / "chat_template.jinja"
        chat_template = template.read_text() if template.is_file() else None
        return cls(
            image_processor=image,
            video_processor=video,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
