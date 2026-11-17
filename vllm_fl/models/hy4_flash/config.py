# SPDX-License-Identifier: Apache-2.0
"""Flash configuration, reusing the existing HY4 config implementation.

The checkpoint uses a nested text_config and a HYV4FlashProcessor name while
its image/video processor schema is the HYV4 VL schema.  Keep those names at
the registration boundary instead of forking the vision processor.
"""

from copy import deepcopy

from vllm_fl.models.hy4.config import HYV4Config
from vllm_fl.models.hy4.vl_config import HYV4VLConfig, HYV4VLVisionConfig

SPARSE_ATTENTION = "deepseek_sparse_attention"
SLIDING_ATTENTION = "sliding_attention"
_SPARSE_ALIASES = {"sparse", "sparse_attention", "indexed_attention", SPARSE_ATTENTION}


class HYV4FlashConfig(HYV4Config):
    model_type = "hy_v4_flash"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_types = [
            SPARSE_ATTENTION if kind in _SPARSE_ALIASES else kind
            for kind in self.layer_types
        ]
        self.indexer_types = self.indexer_types or ["full"] * self.num_hidden_layers
        for name in ("num_attention_heads", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"):
            if getattr(self, f"swa_{name}", None) is None:
                setattr(self, f"swa_{name}", getattr(self, name))
        self.validate_flash_config()

    def rope_for_layer(self, layer_idx):
        rope = deepcopy(self.rope_parameters or {})
        parameters = rope.get(self.layer_types[layer_idx], rope)
        result = {k: deepcopy(v) for k, v in parameters.items() if not isinstance(v, dict)}
        for name in ("mrope_section", "mrope_interleaved"):
            if name in rope:
                result.setdefault(name, deepcopy(rope[name]))
        return result

    def validate_flash_config(self):
        if any(kind not in (SPARSE_ATTENTION, SLIDING_ATTENTION) for kind in self.layer_types):
            raise ValueError("HYV4 Flash requires sparse or sliding attention layers")
        if not getattr(self, "use_compressed_indexer", True):
            raise ValueError("HYV4 Flash requires compressed IndexPool")
        if not getattr(self, "compressed_indexer_skip_fine_select", True) or getattr(self, "compressed_indexer_reuse_fine_indexer", False):
            raise ValueError("Only independent coarse-only IndexPool is supported")
        if getattr(self, "compressed_indexer_key_pooling", "gated") != "gated":
            raise ValueError("Only gated IndexPool is supported")
        ratio = getattr(self, "compressed_indexer_compress_ratio", 4)
        group_topk = getattr(self, "compressed_indexer_group_topk", 512)
        candidate_topk = getattr(self, "compressed_indexer_candidate_topk", self.index_topk)
        if group_topk * ratio != self.index_topk:
            raise ValueError("IndexPool group_topk * ratio must equal index_topk")
        if self.index_topk != candidate_topk:
            raise ValueError("IndexPool candidate_topk must equal index_topk")


class HYV4FlashVLConfig(HYV4VLConfig):
    model_type = "hy_v4_flash"
    default_architecture = "HYV4FlashForConditionalGeneration"
    sub_configs = {"text_config": HYV4FlashConfig, "vision_config": HYV4VLVisionConfig}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.architectures = [self.default_architecture]


__all__ = ["HYV4FlashConfig", "HYV4FlashVLConfig", "SPARSE_ATTENTION", "SLIDING_ATTENTION"]
