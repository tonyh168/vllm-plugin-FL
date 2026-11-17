# SPDX-License-Identifier: Apache-2.0
"""V2-only n-gram hash for Flash Engram; no slot cache is needed."""

import torch
from torch import nn
from vllm.triton_utils import tl, triton


@triton.jit
def _hash_ids_kernel(
    input_ids, token_map, positions, query_start_loc, multipliers, primes, offsets,
    lookback_token_ids, output, num_tokens, num_query_rows,
    input_stride, position_stride, query_stride,
    MAX_NGRAM: tl.constexpr, num_heads, BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr, pad_id, lookback_depth, lookback_row_stride,
):
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    layer = tl.program_id(1)
    num_layers = tl.num_programs(1)
    valid = token < num_tokens
    lo = tl.full((BLOCK_T,), 0, tl.int32)
    hi = tl.full((BLOCK_T,), num_query_rows, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(query_start_loc + (mid + 1) * query_stride, lo < hi, other=0)
        right = token >= end
        active = lo < hi
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    req = tl.minimum(lo, num_query_rows - 1).to(tl.int64)
    chunk_idx = tl.load(query_start_loc + req * query_stride)
    chunk_idx = tl.minimum(chunk_idx, num_tokens - 1).to(tl.int64)
    chunk_start = tl.load(positions + chunk_idx * position_stride)
    position = tl.load(positions + token * position_stride, valid, other=0).to(tl.int64)
    head = tl.arange(0, BLOCK_H)
    blocked = tl.full((BLOCK_T,), False, tl.int1)
    rolling = tl.full((BLOCK_T,), 0, tl.int64)
    for shift in tl.static_range(MAX_NGRAM):
        lookback = position - shift
        in_batch = lookback >= chunk_start
        batch_idx = tl.maximum(token - shift, 0)
        batch_token = tl.load(input_ids + batch_idx * input_stride, valid & in_batch, other=0)
        source = tl.load(token_map + batch_token, valid & in_batch, other=0)
        col = chunk_start - 1 - lookback
        in_window = valid & ~in_batch & (col >= 0) & (col < lookback_depth)
        col = tl.minimum(tl.maximum(col, 0), lookback_depth - 1)
        window_token = tl.load(lookback_token_ids + req * lookback_row_stride + col,
                               in_window, other=-1)
        known = in_window & (window_token >= 0)
        window_source = tl.load(token_map + window_token, known, other=0)
        source = tl.where(in_batch, source, tl.where(known, window_source, pad_id)).to(tl.int64)
        blocked |= (lookback < 0) | ~valid
        value = tl.where(blocked, pad_id, source)
        multiplier = tl.load(multipliers + layer * MAX_NGRAM + shift)
        rolling ^= value * multiplier
        if shift > 0:
            col = (shift - 1) * num_heads + head
            param_offset = layer * (MAX_NGRAM - 1) * num_heads + col
            prime = tl.load(primes + param_offset, head < num_heads, other=1)
            offset = tl.load(offsets + param_offset, head < num_heads, other=0)
            hashed = rolling[:, None] % prime[None, :] + offset[None, :]
            out_offset = (token.to(tl.int64) * num_layers + layer)[:, None] * ((MAX_NGRAM - 1) * num_heads) + col[None, :]
            tl.store(output + out_offset, hashed, valid[:, None] & (head < num_heads))


class NgramHashState(nn.Module):
    block_size = 1
    use_slot_cache = False

    def forward(self, input_ids, positions, query_start_loc, dead_mask,
                lookback_token_ids, lookback_dead_mask, slot_mapping=None, block_table=None):
        num_tokens = input_ids.shape[0]
        num_layers, max_ngram = self.multipliers.shape
        num_heads = self.primes.shape[-1]
        output = input_ids.new_empty((num_tokens, num_layers, (max_ngram - 1) * num_heads), dtype=torch.int32)
        if not num_tokens:
            return output
        assert lookback_token_ids is not None, "Flash Engram requires V2 lookback tokens"
        _hash_ids_kernel[(triton.cdiv(num_tokens, 32), num_layers)](
            input_ids, self.token_map, positions, query_start_loc,
            self.multipliers, self.primes, self.offsets, lookback_token_ids,
            output, num_tokens, query_start_loc.numel() - 1,
            input_ids.stride(0), positions.stride(0), query_start_loc.stride(0),
            MAX_NGRAM=max_ngram, num_heads=num_heads, BLOCK_T=32,
            BLOCK_H=triton.next_power_of_2(num_heads), pad_id=self.pad_id,
            lookback_depth=lookback_token_ids.shape[1],
            lookback_row_stride=lookback_token_ids.stride(0), num_warps=4,
        )
        return output
