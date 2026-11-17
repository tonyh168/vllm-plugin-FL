# SPDX-License-Identifier: Apache-2.0
"""Register HYV4 Flash config, model, processor and checkpoint loader."""

import logging

logger = logging.getLogger(__name__)


def register_hy_v4_flash_support() -> bool:
    from vllm.model_executor import model_loader
    from vllm.model_executor.models import registry as model_registry
    from vllm.transformers_utils import config as transformers_config
    from vllm.transformers_utils import model_arch_config_convertor

    from vllm_fl.models.hy4.config_convertor import HYV4ModelArchConfigConvertor
    from .config import HYV4FlashConfig, HYV4FlashVLConfig
    from .loader import HYV4FlashSafetensorsLoader

    transformers_config._CONFIG_REGISTRY["hy_v4_flash"] = HYV4FlashVLConfig
    model_arch_config_convertor.MODEL_ARCH_CONFIG_CONVERTORS["hy_v4_flash"] = HYV4ModelArchConfigConvertor
    model_registry.ModelRegistry.register_model(
        "HYV4FlashForConditionalGeneration",
        "vllm_fl.models.hy4_flash.vl_model:HYV4FlashForConditionalGeneration",
    )
    model_registry.ModelRegistry.register_model(
        "HYV4FlashForCausalLM", "vllm_fl.models.hy4_flash.model:HYV4FlashForCausalLM",
    )
    from vllm.transformers_utils import processors as vllm_processors
    for name in ("HYV4FlashProcessor", "HYV4VLProcessor", "HYV4VLImageProcessor", "HYV4VLVideoProcessor"):
        vllm_processors._CLASS_TO_MODULE[name] = "vllm_fl.models.hy4.mm_processor"
    model_loader.register_model_loader("hy4_flash_safetensors")(HYV4FlashSafetensorsLoader)
    logger.info("Registered HYV4 Flash runtime components")
    return True
