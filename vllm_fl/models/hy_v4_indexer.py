# SPDX-License-Identifier: Apache-2.0
"""BF16 lightning indexer for HYV4 on PPU.

The HYV4 W8A8 checkpoint keeps every indexer parameter in BF16. T-Head PPU
does not support vLLM's CUDA FP8/DeepGEMM indexer path, so this module keeps
the indexer query, key and paged cache in BF16 and uses regular PyTorch ops.
Those ops are dispatched to the PPU runtime/FlagGems by vllm-plugin-FL.
"""

from __future__ import annotations

import os
import torch
from flag_gems.fused import bf16_paged_mqa_logits
from torch import nn

from vllm.forward_context import get_forward_context
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.utils.torch_utils import _resolve_layer_name
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton

from vllm_fl.ops.bf16_indexer_cache import bf16_indexer_cache_write
from vllm_fl.ops.native_topk import native_topk


_INDEXER_TOPK_MODE = os.environ.get(
    "VLLM_FL_HYV4_INDEXER_TOPK_MODE", "scoped_native"
)
if _INDEXER_TOPK_MODE not in {"original_flaggems", "scoped_native"}:
    raise ValueError(f"unsupported HYV4 indexer topk mode: {_INDEXER_TOPK_MODE}")


def _paged_sequence(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Gather one request's BF16 indexer keys in request-token order."""
    if seq_len <= 0:
        return kv_cache.new_empty((0, kv_cache.shape[-1]))
    block_size = kv_cache.shape[1]
    num_blocks = (seq_len + block_size - 1) // block_size
    blocks = block_table[:num_blocks].to(torch.long)
    offsets = torch.arange(seq_len, device=kv_cache.device, dtype=torch.long)
    slots = blocks.index_select(0, offsets // block_size) * block_size
    slots = slots + offsets.remainder(block_size)
    return kv_cache.reshape(-1, kv_cache.shape[-1]).index_select(0, slots)


def _select_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    topk: int,
    output: torch.Tensor,
) -> None:
    """Select request-local key positions for one query token."""
    output.fill_(-1)
    if keys.shape[0] == 0:
        return

    # HYV4 lightning-indexer score: weighted sum of per-head ReLU(QK).
    # Keep the non-linearity before the head reduction; moving the reduction
    # ahead of ReLU changes the selected tokens when individual heads disagree.
    scores = torch.matmul(keys, q.transpose(0, 1))
    logits = (torch.relu(scores) * weights.to(scores.dtype).unsqueeze(0)).sum(
        dim=-1
    )
    count = min(topk, keys.shape[0])
    indices = torch.topk(logits, count, dim=-1, sorted=True).indices
    output[:count].copy_(indices.to(output.dtype))


class PPUBF16SparseAttnIndexer(nn.Module):
    """Paged BF16 indexer cache and top-k implementation for T-Head PPU."""

    def __init__(
        self,
        *,
        head_dim: int,
        topk_tokens: int,
        cache_config,
        topk_indices_buffer: torch.Tensor,
        prefix: str,
        max_model_len: int,
    ) -> None:
        super().__init__()
        if cache_config is None:
            raise ValueError("HYV4 BF16 indexer requires cache_config")
        self.head_dim = head_dim
        self.topk_tokens = topk_tokens
        self.topk_indices_buffer = topk_indices_buffer
        self.max_model_len = max_model_len
        if head_dim != 128:
            raise ValueError("FlagGems BF16 paged indexer requires head_dim=128")
        if cache_config.block_size != 64:
            raise ValueError("FlagGems BF16 paged indexer requires block_size=64")
        self.register_buffer(
            "_decode_positions",
            torch.arange(max_model_len, dtype=torch.int32),
            persistent=False,
        )
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=head_dim,
            dtype=torch.bfloat16,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )

    def _write_cache(
        self,
        keys: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        num_tokens = slot_mapping.shape[0]
        bf16_indexer_cache_write(keys[:num_tokens], self.k_cache.kv_cache, slot_mapping)

    def _prefill(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        metadata: DeepseekV32IndexerMetadata,
    ) -> None:
        assert metadata.prefill is not None
        # Shape check: ensure num_heads matches FlagGems kernel requirements
        # (prefill doesn't use FlagGems kernel, but check for consistency)
        num_heads = q.shape[-2] if q.ndim >= 2 else 0
        if num_heads not in (32, 64):
            raise ValueError(
                f"Indexer requires num_heads in {{32, 64}}, got {num_heads}. "
                f"Query shape: {q.shape}"
            )
        for chunk in metadata.prefill.chunks:
            cu_seq_lens = chunk.cu_seq_lens.tolist()
            gathered = [
                _paged_sequence(
                    self.k_cache.kv_cache,
                    chunk.block_table[request_id],
                    cu_seq_lens[request_id + 1] - cu_seq_lens[request_id],
                )
                for request_id in range(chunk.num_reqs)
            ]
            keys = torch.cat(gathered, dim=0)
            starts = chunk.cu_seqlen_ks.tolist()
            ends = chunk.cu_seqlen_ke.tolist()
            sequence_starts = chunk.cu_seq_lens[:-1]
            sequence_ids = torch.searchsorted(
                chunk.cu_seq_lens[1:], chunk.cu_seqlen_ks, right=True
            ).tolist()

            for local_row, (start, end, sequence_id) in enumerate(
                zip(starts, ends, sequence_ids, strict=True)
            ):
                token_row = chunk.token_start + local_row
                request_start = int(sequence_starts[sequence_id].item())
                _select_topk(
                    q[token_row],
                    weights[token_row],
                    keys[start:end],
                    self.topk_tokens,
                    self.topk_indices_buffer[token_row],
                )
                valid = self.topk_indices_buffer[token_row] >= 0
                self.topk_indices_buffer[token_row, valid] += start - request_start

    def _decode(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        metadata: DeepseekV32IndexerMetadata,
    ) -> None:
        assert metadata.decode is not None
        # FlagGems bf16_paged_mqa_logits only supports H=32 or H=64 (hardcoded kernels)
        num_heads = q.shape[-2] if q.ndim >= 2 else 0
        if num_heads not in (32, 64):
            raise ValueError(
                f"FlagGems BF16 paged MQA logits requires num_heads in {{32, 64}}, "
                f"got {num_heads}. Query shape: {q.shape}"
            )
        decode = metadata.decode
        decode_lens = decode.decode_lens
        num_decode_tokens = metadata.num_decode_tokens
        if decode.requires_padding:
            padded_q = pack_seq_triton(
                q[:num_decode_tokens], decode_lens, pad_value=0
            )
            padded_weights = pack_seq_triton(
                weights[:num_decode_tokens], decode_lens, pad_value=0
            )
        else:
            padded_q = q[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q.shape[1:]
            )
            padded_weights = weights[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, weights.shape[-1]
            )

        batch_size, next_n = padded_q.shape[:2]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode.seq_lens[:batch_size]
        if seq_lens.ndim == 1:
            seq_lens = seq_lens.unsqueeze(-1)

        logits = bf16_paged_mqa_logits(
            padded_q,
            self.k_cache.kv_cache.unsqueeze(-2),
            padded_weights.reshape(num_padded_tokens, -1),
            seq_lens,
            decode.block_table,
            decode.schedule_metadata,
            max_context_len=self.max_model_len,
            clean_logits=False,
        )
        flat_seq_lens = seq_lens.reshape(-1)[:num_padded_tokens]
        logits.masked_fill_(
            self._decode_positions.unsqueeze(0) >= flat_seq_lens.unsqueeze(1),
            float("-inf"),
        )
        count = min(self.topk_tokens, self.max_model_len)
        # Keep this optimization local to the Indexer.  Globally blacklisting
        # aten::topk also changes grouped MoE router tie ordering and regresses
        # model outputs.  The saved native handle bypasses only FlagGems'
        # pathological 50K two-stage topk implementation.
        if _INDEXER_TOPK_MODE == "original_flaggems":
            topk_indices = torch.topk(logits, count, dim=-1, sorted=True).indices
        else:
            topk_indices = native_topk(
                logits, count, dim=-1, largest=True, sorted=True
            )[1]
        topk_indices = topk_indices.to(self.topk_indices_buffer.dtype)
        # torch.topk still returns concrete indices for the -inf tail when
        # count exceeds a request's current sequence length.  Those indices
        # must remain -1: the sparse MLA conversion uses nonnegative entries
        # to derive topk_length, and treating future/unallocated cache slots as
        # valid dilutes short-context decode attention (for example 185 real
        # tokens were previously reported as 2048).
        valid_counts = torch.minimum(
            flat_seq_lens,
            flat_seq_lens.new_full(flat_seq_lens.shape, count),
        )
        output_positions = self._decode_positions[:count].unsqueeze(0)
        topk_indices.masked_fill_(
            output_positions >= valid_counts.unsqueeze(1),
            -1,
        )

        if decode.requires_padding:
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, next_n, count), decode_lens
            )
        self.topk_indices_buffer[
            : topk_indices.shape[0], :count
        ].copy_(topk_indices)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        del hidden_states
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            self.topk_indices_buffer[: q.shape[0]].fill_(-1)
            return self.topk_indices_buffer

        layer_name = _resolve_layer_name(self.k_cache.prefix)
        metadata = attn_metadata[layer_name]
        if not isinstance(metadata, DeepseekV32IndexerMetadata):
            raise TypeError(
                f"HYV4 BF16 indexer expected DeepseekV32IndexerMetadata, "
                f"got {type(metadata).__name__}"
            )

        self.topk_indices_buffer[: q.shape[0]].fill_(-1)
        self._write_cache(k, metadata.slot_mapping)
        if metadata.num_prefills:
            self._prefill(q, weights, metadata)
        if metadata.num_decodes:
            self._decode(q, weights, metadata)
        return self.topk_indices_buffer


__all__ = ["PPUBF16SparseAttnIndexer", "_paged_sequence", "_select_topk"]
