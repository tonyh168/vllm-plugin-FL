# SPDX-License-Identifier: Apache-2.0
"""Eager BF16 DSA indexing for the PPU pilot, using public FlagGems APIs.

This deliberately retains BF16 queries and keys instead of pretending that
FP32 output from the non-FP8 platform guard is an FP8 cache representation.
It is a different indexer precision from vLLM's FP8 path and needs its own
numerical and model evidence. No quantized checkpoint or graph path is enabled.
"""

from dataclasses import dataclass
import os

import numpy as np
import torch

_PPU_PIECEWISE = os.environ.get("VLLM_FL_PPU_BF16_PIECEWISE_GRAPH", "0") == "1"
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend


@dataclass
class Bf16IndexerMetadata(AttentionMetadata):
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    request_ids: torch.Tensor
    context_lens: torch.Tensor
    num_actual_tokens: int
    max_seq_len: int


class Bf16IndexerMetadataBuilder(AttentionMetadataBuilder[Bf16IndexerMetadata]):
    _cudagraph_support = AttentionCGSupport.NEVER

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        return AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        if kv_cache_spec.block_size != 64:
            raise ValueError("PPU BF16 indexer requires 64-token pages")

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        meta = common_attn_metadata
        starts = np.asarray(meta.query_start_loc_cpu, dtype=np.int32)
        lengths = np.diff(starts)
        ids = np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)
        offsets = np.arange(ids.size, dtype=np.int32) - starts[ids] + 1
        request_ids = torch.from_numpy(ids).to(self.device, non_blocking=True)
        query_lengths = torch.from_numpy(lengths).to(self.device, non_blocking=True)
        offsets = torch.from_numpy(offsets).to(self.device, non_blocking=True)
        contexts = meta.seq_lens[request_ids] - query_lengths[request_ids] + offsets
        return Bf16IndexerMetadata(
            slot_mapping=meta.slot_mapping,
            block_table=meta.block_table_tensor,
            request_ids=request_ids,
            context_lens=contexts.to(torch.int32),
            num_actual_tokens=meta.num_actual_tokens,
            max_seq_len=meta.max_seq_len,
        )


class Bf16IndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_name():
        return "DEEPSEEK_V32_PPU_BF16_INDEXER"

    @classmethod
    def get_supported_head_sizes(cls):
        return [128]

    @staticmethod
    def get_builder_cls():
        return Bf16IndexerMetadataBuilder


# Amortize Python dispatch across prefill queries while bounding the score tile
# to 8 MiB, including the separately validated long-context evaluation path.
_INDEXER_QUERY_CHUNK_TOKENS = 256


def _query_chunk_tokens(max_context_len):
    """Keep the F32 score tile within the original 8 MiB workspace."""
    if not 1 <= max_context_len <= 65536:
        raise ValueError("PPU BF16 indexer context must be within 1..65536")
    return min(_INDEXER_QUERY_CHUNK_TOKENS, (8 * 1024 * 1024) // (4 * max_context_len))


def paged_topk(q, cache, weights, contexts, block_table, topk, max_context_len):
    """Return logical token indices; invalid and future positions stay -1."""
    from flag_gems import sort as gems_sort
    from flag_gems import topk as gems_topk
    from flaggems_vllm import bf16_paged_mqa_logits

    from .register_prefix_radix import persistent_topk_indices, supports

    if q.dtype != torch.bfloat16 or q.shape[1:] != (64, 128):
        raise ValueError("PPU BF16 indexer requires BF16 [tokens,64,128] queries")
    if cache.dtype != torch.bfloat16 or cache.shape[1:] != (64, 128):
        raise ValueError("PPU BF16 indexer requires BF16 [pages,64,128] cache")
    if not cache.is_contiguous():
        raise ValueError("Public BF16 paged MQA requires dense physical cache pages")
    # topk specializes on both its input width and k. Keep these stable across
    # decode steps within a physical page, while context_lens still masks every
    # future position to -inf. This avoids a new JIT specialization per token.
    max_context_len = ((max_context_len + 63) // 64) * 64
    output = torch.full((q.shape[0], topk), -1, dtype=torch.int32, device=q.device)
    width = min(topk, max_context_len)
    if width == 0:
        return output
    # Bound the logits allocation even for large prefill batches.
    query_chunk = _query_chunk_tokens(max_context_len)
    for start in range(0, q.shape[0], query_chunk):
        end = min(start + query_chunk, q.shape[0])
        logits = bf16_paged_mqa_logits(
            q[start:end].contiguous().unsqueeze(1),
            cache.unsqueeze(2),
            weights[start:end].float().contiguous(),
            contexts[start:end].reshape(-1, 1).contiguous(),
            block_table[start:end].contiguous(),
            None,
            max_context_len,
            clean_logits=True,
        )
        if max_context_len > topk and supports(logits, width):
            output[start:end, :width] = persistent_topk_indices(logits, width)
            continue
        if max_context_len > topk:
            # PPU compilation of a monolithic bitonic TopK above 2048 keys
            # exceeds the engine RPC deadline. Public radix sort has bounded
            # tiles and returns the same largest finite key set.
            values, indices = gems_sort(logits, dim=-1, descending=True)
            values, indices = values[:, :width], indices[:, :width]
        else:
            values, indices = gems_topk(logits, width, dim=-1, sorted=False)
        output[start:end, :width] = torch.where(torch.isfinite(values), indices, -1).to(
            torch.int32
        )
    return output


def indexer_forward(self, hidden_states, qr, positions, rotary_emb):
    from flaggems_vllm import qsa_store_cache_rows

    q, _ = self.wq_b(qr)
    q = q.view(-1, self.n_head, self.head_dim)
    q_pe, q_nope = torch.split(q, [self.rope_dim, self.head_dim - self.rope_dim], -1)
    projected, _ = self.wk_weights_proj(hidden_states)
    k = self.k_norm(projected[:, : self.head_dim])
    weights = projected[:, self.head_dim :]
    k_pe, k_nope = torch.split(k, [self.rope_dim, self.head_dim - self.rope_dim], -1)
    q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
    q = torch.cat([q_pe.reshape_as(q[..., : self.rope_dim]), q_nope], -1)
    k = torch.cat([k_pe.reshape(-1, self.rope_dim), k_nope], -1)
    weights = weights.float() * self.softmax_scale * self.n_head**-0.5

    if _PPU_PIECEWISE:
        torch.ops.vllm_fl.ppu_bf16_indexer(
            q, k, weights, self.k_cache.kv_cache, self.topk_indices_buffer,
            self.k_cache.prefix, self.topk_tokens, self.max_model_len,
        )
        return self.topk_indices_buffer

    self.topk_indices_buffer[: hidden_states.shape[0]].fill_(-1)
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):
        # Account for the bounded logits workspace during the profile run.
        _ = torch.empty(
            (_query_chunk_tokens(self.max_model_len), self.max_model_len),
            dtype=torch.float32,
            device=q.device,
        )
        return self.topk_indices_buffer
    meta = metadata[self.k_cache.prefix]
    if not isinstance(meta, Bf16IndexerMetadata):
        raise TypeError("PPU BF16 indexer received incompatible metadata")
    count = meta.num_actual_tokens
    cache = self.k_cache.kv_cache
    qsa_store_cache_rows(
        cache.unsqueeze(2), meta.slot_mapping[:count], k[:count].contiguous()
    )
    tables = meta.block_table.index_select(0, meta.request_ids[:count].long())
    selected = paged_topk(
        q[:count],
        cache,
        weights[:count],
        meta.context_lens[:count],
        tables,
        self.topk_tokens,
        meta.max_seq_len,
    )
    self.topk_indices_buffer[:count].copy_(selected)
    return self.topk_indices_buffer
