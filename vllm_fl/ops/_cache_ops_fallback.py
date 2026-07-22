# Copyright (c) 2026 BAAI. All rights reserved.
#
# Pure-PyTorch fallback implementations for _C_cache_ops kernels.
# Used on platforms (e.g. Hygon) where vLLM is built with
# VLLM_TARGET_DEVICE=empty and no vendor library (like mcoplib)
# provides native implementations.

import torch


def concat_and_cache_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    """
    Concatenate kv_c and k_pe, then write into paged kv_cache.

    Args:
        kv_c: [num_tokens, kv_lora_rank]
        k_pe: [num_tokens, pe_dim]
        kv_cache: [num_blocks, block_size, kv_lora_rank + pe_dim]
        slot_mapping: [num_actual_tokens] (may be shorter than kv_c due to padding)
        kv_cache_dtype: cache dtype string (e.g. "auto")
        scale: scale tensor for fp8 quantization (unused for "auto")
    """
    num_tokens = slot_mapping.size(0)
    block_size = kv_cache.size(1)

    kv_c_slice = kv_c[:num_tokens]
    k_pe_slice = k_pe[:num_tokens]

    valid_mask = slot_mapping >= 0
    valid_slots = slot_mapping[valid_mask]

    block_idx = valid_slots // block_size
    block_offset = valid_slots % block_size

    combined = torch.cat([kv_c_slice[valid_mask], k_pe_slice[valid_mask]], dim=-1)

    if kv_cache_dtype == "auto":
        kv_cache[block_idx, block_offset] = combined.to(kv_cache.dtype)
    else:
        kv_cache[block_idx, block_offset] = combined.to(kv_cache.dtype)
