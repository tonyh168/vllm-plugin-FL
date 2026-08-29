# SPDX-License-Identifier: Apache-2.0
"""Capture-safe BF16 indexer cache writer for T-Head PPU."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _bf16_indexer_cache_write_kernel(
    keys_ptr,
    cache_ptr,
    slot_mapping_ptr,
    key_stride_token: tl.constexpr,
    key_stride_dim: tl.constexpr,
    cache_stride_slot: tl.constexpr,
    cache_stride_dim: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    token = tl.program_id(0)
    dims = tl.arange(0, BLOCK_DIM)
    slot = tl.load(slot_mapping_ptr + token)
    mask = (slot >= 0) & (dims < HEAD_DIM)
    values = tl.load(
        keys_ptr + token * key_stride_token + dims * key_stride_dim,
        mask=mask,
    )
    tl.store(
        cache_ptr + slot * cache_stride_slot + dims * cache_stride_dim,
        values,
        mask=mask,
    )


def bf16_indexer_cache_write(
    keys: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Write BF16 indexer keys into a token-major paged cache in place.

    ``slot_mapping == -1`` denotes graph padding and is ignored in the Triton
    kernel, avoiding the data-dependent boolean indexing and Python ``bool``
    that cannot be captured by a FULL decode CUDAGraph.
    """
    if keys.ndim != 2 or cache.ndim != 3 or slot_mapping.ndim != 1:
        raise ValueError(
            "bf16_indexer_cache_write expects keys [T,D], cache [B,S,D], "
            "and slot_mapping [T]"
        )
    if keys.dtype != torch.bfloat16 or cache.dtype != torch.bfloat16:
        raise ValueError("BF16 indexer cache writer requires bfloat16 tensors")
    if keys.shape[1] != cache.shape[2]:
        raise ValueError("indexer key/cache head dimensions must match")
    if keys.shape[0] < slot_mapping.shape[0]:
        raise ValueError("slot_mapping cannot contain more tokens than keys")
    if slot_mapping.numel() == 0:
        return

    head_dim = keys.shape[1]
    block_dim = triton.next_power_of_2(head_dim)
    _bf16_indexer_cache_write_kernel[(slot_mapping.shape[0],)](
        keys,
        cache,
        slot_mapping,
        key_stride_token=keys.stride(0),
        key_stride_dim=keys.stride(1),
        cache_stride_slot=cache.stride(1),
        cache_stride_dim=cache.stride(2),
        HEAD_DIM=head_dim,
        BLOCK_DIM=block_dim,
    )


__all__ = ["bf16_indexer_cache_write"]
