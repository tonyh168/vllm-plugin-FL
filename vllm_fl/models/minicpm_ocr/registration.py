# SPDX-License-Identifier: Apache-2.0
"""Register OCR support using the existing vLLM and Transformers registries."""


def register_minicpm_ocr_support() -> bool:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    # General plugins run for every model. Do not require newer Transformers
    # OCR dependencies when serving an unrelated architecture.
    if "minicpmv4_6" not in CONFIG_MAPPING_NAMES:
        return False

    from transformers import AutoConfig, AutoProcessor, AutoVideoProcessor

    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    from .config import MiniCPMOcrConfig
    from .processor import MiniCPMOcrProcessor
    from .video_processing_minicpm_ocr import MiniCPMOcrVideoProcessor

    _CONFIG_REGISTRY.setdefault("minicpm_ocr", MiniCPMOcrConfig)
    AutoConfig.register("minicpm_ocr", MiniCPMOcrConfig, exist_ok=True)
    AutoProcessor.register(MiniCPMOcrConfig, MiniCPMOcrProcessor, exist_ok=True)
    AutoVideoProcessor.register(
        MiniCPMOcrConfig, MiniCPMOcrVideoProcessor, exist_ok=True
    )
    architecture = "MiniCPMOcrForConditionalGeneration"
    if architecture not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            architecture,
            "vllm_fl.models.minicpm_ocr.model:MiniCPMOcrForConditionalGeneration",
        )

    return True
