# SPDX-License-Identifier: Apache-2.0
"""Per-layer Flash sparse/window MLA on the portable HY4 MetaX backend."""

from copy import deepcopy

import torch
from torch import nn

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope

from vllm_fl.models.hy4.model import HYV4Attention

from .compressed_indexer import FlashCompressedIndexer
from .config import SLIDING_ATTENTION
from .swa import FlashWindowIndexer


class HYV4FlashAttention(HYV4Attention):
    def __init__(self, vllm_config, config, layer_idx, prefix, topk_indices_buffer, runtime_plan):
        self.is_window_layer = config.layer_types[layer_idx] == SLIDING_ATTENTION
        config = deepcopy(config)
        config.rope_parameters = config.rope_for_layer(layer_idx)
        heads = config.swa_num_attention_heads if self.is_window_layer else config.num_attention_heads
        nope = config.swa_qk_nope_head_dim if self.is_window_layer else config.qk_nope_head_dim
        rope = config.swa_qk_rope_head_dim if self.is_window_layer else config.qk_rope_head_dim
        value = config.swa_v_head_dim if self.is_window_layer else config.v_head_dim
        # Sliding layers have full sparse-MLA top-k semantics (a causal window),
        # not the parent's dense-MLA-without-indexer path.
        # Flash has no shared indexers: every sparse and sliding layer owns
        # its top-k producer.  Do not inherit the sparse-only layer filter.
        config.indexer_types = ["full"] * config.num_hidden_layers
        # Deep-copied config retains all sparse-MLA flags when the parent
        # computes its sparse-layer predicate.
        sparse_layer_types = getattr(config, "_SPARSE_LAYER_TYPES", None)
        if sparse_layer_types is None:
            from vllm_fl.models.hy4 import attention as hy4_attention
            hy4_attention._SPARSE_LAYER_TYPES = set(hy4_attention._SPARSE_LAYER_TYPES) | {SLIDING_ATTENTION}
        super().__init__(
            vllm_config=vllm_config, config=config, hidden_size=config.hidden_size,
            num_heads=heads, qk_nope_head_dim=nope, qk_rope_head_dim=rope,
            v_head_dim=value, q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=vllm_config.model_config.max_model_len,
            cache_config=vllm_config.cache_config, quant_config=vllm_config.quant_config,
            prefix=prefix, topk_indices_buffer=topk_indices_buffer,
            layer_idx=layer_idx, runtime_plan=runtime_plan,
        )
        self.mla_attn.use_sparse = self.is_sparse
        self.mla_attn.sliding_window = None  # full KV history; only top-k is windowed
        # The parent intentionally builds only sparse indexers.  Replace that
        # selection with Flash's independent IndexPool and causal SWA indexer.
        self.is_sparse = True
        self.skip_topk = False
        self.create_indexer = True
        if self.is_window_layer:
            self.indexer = FlashWindowIndexer(config.sliding_window, topk_indices_buffer, f"{prefix}.attn")
        else:
            self.indexer = FlashCompressedIndexer(
                vllm_config, config, topk_indices_buffer, f"{prefix}.indexer", runtime_plan,
            )
        self.indexer_rope_emb = get_rope(
            rope, max_position=vllm_config.model_config.max_model_len,
            rope_parameters=config.rope_parameters, is_neox_style=False,
        )
        self.mla_attn.indexer = self.indexer
        self.mla_attn.is_sparse = True
        self.mla_attn.use_sparse = True
        self.mla_attn.sliding_window = None  # full KV history; only top-k is windowed
        self.post_proj_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, llama_4_scaling=None):
        return self.post_proj_norm(super().forward(positions, hidden_states, llama_4_scaling))
