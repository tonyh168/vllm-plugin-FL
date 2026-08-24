# Copyright (c) 2026 BAAI. All rights reserved.

"""
Torch-based fused MoE implementation for MetaX (MACA) platform.

MACA's triton compiler does not support tl.dot(int8, int8), so this
implementation dequantizes INT8 W8A8 tensors to bf16 before matmul,
using pure PyTorch ops that work on any CUDA-alike backend.
"""

from typing import Any, Optional

import torch


def invoke_fused_moe_torch_int8(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: Optional[torch.Tensor],
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: Any,
    use_int8_w8a8: bool,
    per_channel_quant: bool,
    block_shape: Optional[list[int]] = None,
    B_bias: Optional[torch.Tensor] = None,
):
    """
    Pure-torch fused MoE kernel for INT8 W8A8 quantized models.

    This mirrors the triton fused_moe_kernel logic but uses torch matmul
    after dequantizing int8 tensors to bfloat16.

    Args:
        A: [num_tokens * top_k, K] int8 activations (pre-sorted by expert)
        B: [E, N, K] int8 expert weights
        C: [num_tokens * top_k, E, N] bf16 output (pre-allocated, to be written)
           OR [num_tokens, top_k, N] depending on layout
        A_scale: per-token or per-block activation scale
        B_scale: per-expert or per-block weight scale
        topk_weights: [num_tokens, top_k] routing weights
        sorted_token_ids: [EM] sorted token indices (padded to BLOCK_SIZE_M)
        expert_ids: [num_blocks] expert id for each block of sorted_token_ids
        num_tokens_post_padded: [1] actual number of valid token slots
        mul_routed_weight: whether to multiply by topk_weights
        top_k: number of experts per token
        config: dict with BLOCK_SIZE_M etc.
        compute_type: target compute dtype (unused, always bf16)
        use_int8_w8a8: must be True
        per_channel_quant: whether B_scale is per-channel
        block_shape: [group_k, group_n] for block-wise quantization
        B_bias: optional per-expert bias [E, N]
    """
    assert use_int8_w8a8, "This implementation only handles INT8 W8A8"

    BLOCK_SIZE_M = config.get("BLOCK_SIZE_M", 64)
    num_valid = num_tokens_post_padded.item() if num_tokens_post_padded.numel() == 1 else num_tokens_post_padded[0].item()
    M_total = A.size(0)
    K = A.size(1)
    E = B.size(0)
    N = B.size(1)

    out_dtype = C.dtype

    num_blocks = expert_ids.size(0)

    for block_idx in range(num_blocks):
        start = block_idx * BLOCK_SIZE_M
        if start >= num_valid:
            break

        end = min(start + BLOCK_SIZE_M, num_valid)
        token_ids = sorted_token_ids[start:end]

        # Filter out padding tokens (id >= M_total)
        valid_mask = token_ids < M_total
        if not valid_mask.any():
            continue

        valid_token_ids = token_ids[valid_mask]
        expert_id = expert_ids[block_idx].item()

        # Gather activations: [block_tokens, K] int8
        a_block = A[valid_token_ids]  # int8
        # Expert weight: [N, K] int8
        b_expert = B[expert_id]  # [N, K] int8

        # Dequantize and matmul
        if block_shape is not None and block_shape[0] > 0 and block_shape[1] > 0:
            # Block-wise quantization: dequant per block then accumulate
            group_k = block_shape[0]
            group_n = block_shape[1]
            n_k_blocks = (K + group_k - 1) // group_k
            n_n_blocks = (N + group_n - 1) // group_n

            acc = torch.zeros(
                (valid_token_ids.size(0), N), dtype=torch.float32, device=A.device
            )

            for ki in range(n_k_blocks):
                k_start = ki * group_k
                k_end = min(k_start + group_k, K)
                a_slice = a_block[:, k_start:k_end].to(torch.float32)
                b_slice = b_expert[:, k_start:k_end].to(torch.float32)

                # a_scale: [M_total, n_k_blocks] or [num_tokens*topk, n_k_blocks]
                if A_scale is not None and A_scale.ndim == 2:
                    a_s = A_scale[valid_token_ids, ki].unsqueeze(1)  # [block, 1]
                elif A_scale is not None:
                    a_s = A_scale[valid_token_ids].unsqueeze(1)
                else:
                    a_s = torch.ones(1, device=A.device)

                # b_scale: [E, n_k_blocks, n_n_blocks] or [E, n_n_blocks]
                if B_scale is not None and B_scale.ndim == 3:
                    b_s = B_scale[expert_id, ki, :]  # [n_n_blocks]
                    # expand to [N]
                    b_s_full = b_s.repeat_interleave(group_n)[:N].unsqueeze(0)
                elif B_scale is not None and B_scale.ndim == 2:
                    b_s = B_scale[expert_id, ki] if B_scale.size(1) > 1 else B_scale[expert_id, 0]
                    b_s_full = b_s.unsqueeze(0)
                else:
                    b_s_full = torch.ones(1, device=A.device)

                # matmul: [block, k_slice] @ [k_slice, N] -> [block, N]
                partial = torch.mm(a_slice, b_slice.T)
                acc += partial * a_s * b_s_full

            result = acc
        else:
            # Per-tensor or per-channel quantization
            a_f = a_block.to(torch.float32)
            b_f = b_expert.to(torch.float32)

            # matmul: [block, K] @ [K, N] -> [block, N]
            raw = torch.mm(a_f, b_f.T)

            # Apply scales
            if per_channel_quant:
                # A_scale: [M_total, 1] or [M_total]
                # B_scale: [E, N]
                if A_scale is not None:
                    a_s = A_scale[valid_token_ids]
                    if a_s.ndim == 1:
                        a_s = a_s.unsqueeze(1)
                    raw = raw * a_s
                if B_scale is not None:
                    b_s = B_scale[expert_id]
                    if b_s.ndim == 1:
                        b_s = b_s.unsqueeze(0)
                    raw = raw * b_s
            else:
                # Per-tensor: A_scale [M_total, 1], B_scale [E, 1]
                if A_scale is not None:
                    a_s = A_scale[valid_token_ids]
                    if a_s.ndim == 1:
                        a_s = a_s.unsqueeze(1)
                    raw = raw * a_s
                if B_scale is not None:
                    b_s = B_scale[expert_id]
                    if b_s.ndim == 0:
                        raw = raw * b_s
                    else:
                        raw = raw * b_s.unsqueeze(0) if b_s.ndim == 1 else raw * b_s

            result = raw

        # Add bias (in float32)
        if B_bias is not None:
            result = result + B_bias[expert_id].to(torch.float32)

        # Apply routing weights (in float32)
        if mul_routed_weight and topk_weights is not None:
            weights = topk_weights.view(-1)[valid_token_ids].unsqueeze(1).to(torch.float32)
            result = result * weights

        # Write back to C (convert to output dtype at the end)
        C[valid_token_ids] = result.to(out_dtype)
