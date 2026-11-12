# SPDX-License-Identifier: Apache-2.0
"""Opaque BF16 indexer metadata/cache boundary for Thead PIECEWISE graph.

Only this existing eager attention tail is outside capture. Numerical work
still calls the same public APIs; cache/output mutation is explicit to Dynamo.
"""
import torch
from vllm.forward_context import get_forward_context


@torch.library.custom_op("vllm_fl::ppu_bf16_indexer", mutates_args=("cache", "output"))
def ppu_bf16_indexer(
    q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor,
    cache: torch.Tensor, output: torch.Tensor,
    prefix: str, topk: int, max_model_len: int,
) -> None:
    from flaggems_vllm import qsa_store_cache_rows
    from .bf16_indexer import Bf16IndexerMetadata, paged_topk, _query_chunk_tokens

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("BF16 dynamic indexer must remain outside device graph capture")
    output[:q.shape[0]].fill_(-1)
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):
        _ = torch.empty((_query_chunk_tokens(max_model_len), max_model_len), dtype=torch.float32, device=q.device)
        return
    meta = metadata[prefix]
    if not isinstance(meta, Bf16IndexerMetadata):
        raise TypeError('PPU BF16 indexer received incompatible metadata')
    count = meta.num_actual_tokens
    qsa_store_cache_rows(cache.unsqueeze(2), meta.slot_mapping[:count], k[:count].contiguous())
    tables = meta.block_table.index_select(0, meta.request_ids[:count].long())
    selected = paged_topk(q[:count], cache, weights[:count], meta.context_lens[:count], tables, topk, meta.max_seq_len)
    output[:count].copy_(selected)
    return



@ppu_bf16_indexer.register_fake
def _fake(q, k, weights, cache, output, prefix, topk, max_model_len):
    # Explicit mutations carry dependencies; the caller returns its own buffer.
    return None
