# Copyright (c) 2026 BAAI. All rights reserved.
"""Qwen3.5-VL sequence-classification (score / reward head) for vLLM.

Motivation
----------
The Jev / OCR-RM-2B checkpoint is a `Qwen3_5ForSequenceClassification`:

    model.visual.*            # Qwen3.5 vision tower
    model.language_model.*    # 24-layer hybrid (GatedDeltaNet + periodic full attn)
    score.weight  [num_labels, hidden_size]   # bias=False head, no lm_head

Scoring is `prob = softmax(score(hidden)[LAST])` — i.e. the pooling position is
the **last** token, not CLS.

vLLM 0.24 already ships every piece this needs (the Qwen3.5-VL backbone, pooling
models, `LAST` sequence pooling, and the `/score` endpoint) but not this
particular combination. This module only composes them, following the same shape
as `vllm/model_executor/models/jina_vl.py` (VLM + `is_pooling_model` + a
classifier head handed to `DispatchPooler.for_seq_cls`).

Nothing in vLLM itself is modified.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
)
from vllm.model_executor.models.interfaces import IntermediateTensors
from vllm.multimodal import MULTIMODAL_REGISTRY


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
@default_pooling_type(seq_pooling_type="LAST", tok_pooling_type="ALL")
class Qwen3_5ForSequenceClassification(Qwen3_5ForConditionalGeneration):
    """Qwen3.5-VL backbone + a `num_labels`-way linear score head, LAST pooling.

    `forward` returns the backbone hidden states; the pooler then applies the
    score head at the last position and the classifier activation (softmax when
    `problem_type == "single_label_classification"`).
    """

    is_pooling_model = True

    # `Qwen3VLForConditionalGeneration.hf_to_vllm_mapper` already maps the two
    # prefixes this checkpoint uses (`model.language_model.` -> `language_model.model.`,
    # `model.visual.` -> `visual.`). `score.` needs no rewriting. Re-declared
    # explicitly so this class documents its own weight contract.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "model.language_model.": "language_model.model.",
            "score.": "score.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        text_config = (
            hf_config.get_text_config()
            if hasattr(hf_config, "get_text_config")
            else hf_config.text_config
        )

        # `num_labels` is absent from the checkpoint's config.json (HF defaults to
        # 2 and the shipped `score.weight` is [2, 2048]). `--hf-overrides
        # '{"num_labels": 2}'` sets it explicitly; this fallback keeps a bare
        # config.json working too.
        self.num_labels = int(getattr(hf_config, "num_labels", 0) or 2)

        # ⚠️ 必须用 `head_dtype`（vLLM 0.24 默认 fp32），**不能**用 `model_config.dtype`。
        # 原因：pooler 的 `ClassifierPoolerHead.forward` 会先把 pooled_data 转成
        # `head_dtype` 再喂给 classifier（seqwise/heads.py:176 `pooled_data.to(self.head_dtype)`）。
        # 若 classifier 权重是 bf16 而 pooled_data 被转成 fp32，Iluvatar 的 Triton
        # `tl.dot` 会直接断言：
        #   AssertionError: Both operands must be same dtype. Got fp32 and bf16
        # （实测 iluvatar-186 / 2026-10-22）。用 head_dtype 后两侧同为 fp32，一致。
        self.score = nn.Linear(
            text_config.hidden_size,
            self.num_labels,
            bias=False,
            dtype=model_config.head_dtype,
        )

        pooler_config = model_config.pooler_config
        assert pooler_config is not None, (
            "pooling model requires a PoolerConfig; start with --runner pooling"
        )
        self.pooler = DispatchPooler.for_seq_cls(
            pooler_config, classifier=self.score
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if not isinstance(hidden_states, torch.Tensor):
            return hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # `mtp.` weights do not exist for this checkpoint (mtp_num_hidden_layers=0)
        # but skipping the prefix keeps the loader quiet for sibling checkpoints.
        loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


__all__ = ["Qwen3_5ForSequenceClassification"]
