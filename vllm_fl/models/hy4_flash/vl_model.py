# SPDX-License-Identifier: Apache-2.0
"""Flash text/vision composition using the existing HYV4 VL processor."""

from functools import cached_property

from vllm.model_executor.models.utils import WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.processor import get_processor_kwargs_keys

from vllm_fl.models.hy4.mm_preprocess import (
    HYV4VLDummyInputsBuilder,
    HYV4VLMultiModalProcessor,
    HYV4VLProcessingInfo,
)
from vllm_fl.models.hy4.mm_processor import HYV4VLProcessor, HYV4VLProcessorKwargs
from vllm_fl.models.hy4.vl_model import HYV4VLForConditionalGeneration

from .config import HYV4FlashVLConfig
from .model import HYV4FlashForCausalLM, validate_runtime


class HYV4FlashVLProcessingInfo(HYV4VLProcessingInfo):
    def get_hf_config(self) -> HYV4FlashVLConfig:
        return self.ctx.get_hf_config(HYV4FlashVLConfig)

    def _load_processor(self, kwargs) -> HYV4VLProcessor:
        dynamic_keys = get_processor_kwargs_keys(HYV4VLProcessorKwargs)
        options = {k: v for k, v in self.ctx.get_merged_mm_kwargs(kwargs).items()
                   if k not in dynamic_keys}
        options["config"] = self.get_hf_config()
        options["tokenizer"] = self.ctx.tokenizer
        return HYV4VLProcessor.from_pretrained(
            self.ctx.model_config.model, revision=self.ctx.model_config.revision,
            **options,
        )

    @cached_property
    def _default_processor(self):
        return self._load_processor({})

    def get_hf_processor(self, **kwargs):
        return self._load_processor(kwargs) if kwargs else self._default_processor


@MULTIMODAL_REGISTRY.register_processor(
    HYV4VLMultiModalProcessor, info=HYV4FlashVLProcessingInfo,
    dummy_inputs=HYV4VLDummyInputsBuilder,
)
class HYV4FlashForConditionalGeneration(HYV4VLForConditionalGeneration):
    language_model_architecture = "HYV4FlashForCausalLM"
    requires_raw_input_tokens = True
    supports_pp = False  # type: ignore[assignment]

    def __init__(self, *, vllm_config, prefix=""):
        validate_runtime(vllm_config)
        # Parent constructs the ViT and resolves language_model_architecture,
        # which is the Flash class here (not the sparse-only HYV4 model).
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.token_lookback_depth = self.language_model.token_lookback_depth
        engram_hash = self.language_model.model.engram_hash
        if engram_hash is not None:
            engram_hash.visual_token_ids = (
                self.config.image_token_id, self.config.video_token_id,
            )

    @staticmethod
    def get_model_state_cls():
        return HYV4FlashForCausalLM.get_model_state_cls()

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, lookback_token_ids=None, **kwargs):
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds,
            lookback_token_ids=lookback_token_ids,
        )
