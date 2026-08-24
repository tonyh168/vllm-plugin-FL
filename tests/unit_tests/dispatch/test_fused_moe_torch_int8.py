# Copyright (c) 2026 BAAI. All rights reserved.
"""
Unit test for torch-based INT8 W8A8 fused MoE implementation.

Tests the invoke_fused_moe_torch_int8 function against a naive
reference implementation to verify numerical correctness.
"""

import pytest
import torch


def _naive_fused_moe_int8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: int,
    mul_routed_weight: bool,
    top_k: int,
    per_channel_quant: bool,
    block_shape: list[int] | None,
    BLOCK_SIZE_M: int,
    B_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Naive reference: loop over each valid token, do full dequant + matmul.
    Returns the expected output tensor.
    """
    M = A.size(0)
    N = B.size(1)
    K = A.size(1)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device=A.device)

    num_blocks = expert_ids.size(0)
    for block_idx in range(num_blocks):
        start = block_idx * BLOCK_SIZE_M
        if start >= num_tokens_post_padded:
            break
        end = min(start + BLOCK_SIZE_M, num_tokens_post_padded)
        token_ids = sorted_token_ids[start:end]

        for i, tid in enumerate(token_ids):
            tid_val = tid.item()
            if tid_val >= M:
                continue
            expert_id = expert_ids[block_idx].item()

            a_row = A[tid_val].to(torch.float32)  # [K]
            b_mat = B[expert_id].to(torch.float32)  # [N, K]

            if block_shape is not None and block_shape[0] > 0 and block_shape[1] > 0:
                group_k = block_shape[0]
                group_n = block_shape[1]
                result = torch.zeros(N, dtype=torch.float32, device=A.device)
                for ki in range((K + group_k - 1) // group_k):
                    k_s = ki * group_k
                    k_e = min(k_s + group_k, K)
                    a_slice = a_row[k_s:k_e]
                    for ni in range((N + group_n - 1) // group_n):
                        n_s = ni * group_n
                        n_e = min(n_s + group_n, N)
                        b_slice = b_mat[n_s:n_e, k_s:k_e]
                        partial = a_slice @ b_slice.T
                        a_s = A_scale[tid_val, ki] if A_scale.ndim == 2 else A_scale[tid_val]
                        b_s = B_scale[expert_id, ki, ni] if B_scale.ndim == 3 else B_scale[expert_id, 0]
                        result[n_s:n_e] += partial * a_s * b_s
                val = result
            elif per_channel_quant:
                raw = a_row @ b_mat.T  # [N]
                a_s = A_scale[tid_val].item() if A_scale[tid_val].numel() == 1 else A_scale[tid_val]
                b_s = B_scale[expert_id]  # [N]
                val = raw * a_s * b_s
            else:
                raw = a_row @ b_mat.T
                a_s = A_scale[tid_val].item() if A_scale[tid_val].numel() == 1 else A_scale[tid_val]
                b_s = B_scale[expert_id].item() if B_scale[expert_id].numel() == 1 else B_scale[expert_id]
                val = raw * a_s * b_s

            if B_bias is not None:
                val = val + B_bias[expert_id].to(torch.float32)

            if mul_routed_weight and topk_weights is not None:
                w = topk_weights.view(-1)[tid_val].item()
                val = val * w

            C[tid_val] = val.to(torch.bfloat16)

    return C


def _make_sorted_token_ids_and_expert_ids(
    num_tokens: int, top_k: int, num_experts: int, block_size_m: int, device: torch.device
):
    """
    Simulate moe_align_block_size output: assign tokens to experts,
    sort by expert, and pad to block_size_m boundaries.
    """
    total_slots = num_tokens * top_k
    # Random expert assignment
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device=device)

    # Count tokens per expert
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    for e in range(num_experts):
        expert_counts[e] = (topk_ids == e).sum()

    # Build sorted_token_ids and expert_ids
    sorted_ids_list = []
    expert_ids_list = []
    flat_topk_ids = topk_ids.view(-1)

    for e in range(num_experts):
        mask = (flat_topk_ids == e)
        token_indices = mask.nonzero(as_tuple=True)[0]
        count = token_indices.size(0)
        if count == 0:
            continue
        # Pad to BLOCK_SIZE_M boundary
        padded_count = ((count + block_size_m - 1) // block_size_m) * block_size_m
        padded_ids = torch.full((padded_count,), total_slots, dtype=torch.int32, device=device)
        padded_ids[:count] = token_indices.to(torch.int32)
        sorted_ids_list.append(padded_ids)

        num_blocks_for_expert = padded_count // block_size_m
        expert_ids_list.extend([e] * num_blocks_for_expert)

    sorted_token_ids = torch.cat(sorted_ids_list) if sorted_ids_list else torch.zeros(0, dtype=torch.int32, device=device)
    expert_ids = torch.tensor(expert_ids_list, dtype=torch.int32, device=device)
    num_tokens_post_padded = torch.tensor([sorted_token_ids.size(0)], dtype=torch.int32, device=device)

    return topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded


class TestFusedMoeTorchInt8:
    """Test suite for invoke_fused_moe_torch_int8."""

    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        pytest.skip("CUDA not available")

    def _run_test(
        self,
        device: torch.device,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        K: int,
        N: int,
        per_channel_quant: bool,
        block_shape: list[int] | None,
        mul_routed_weight: bool,
        use_bias: bool = False,
    ):
        from vllm_fl.dispatch.backends.vendor.metax.impl.fused_moe_torch import (
            invoke_fused_moe_torch_int8,
        )

        BLOCK_SIZE_M = 16
        total_slots = num_tokens * top_k

        # Generate random int8 data
        A = torch.randint(-128, 127, (total_slots, K), dtype=torch.int8, device=device)
        B = torch.randint(-128, 127, (num_experts, N, K), dtype=torch.int8, device=device)

        # Generate scales
        if block_shape is not None and block_shape[0] > 0 and block_shape[1] > 0:
            n_k_blocks = (K + block_shape[0] - 1) // block_shape[0]
            n_n_blocks = (N + block_shape[1] - 1) // block_shape[1]
            A_scale = torch.rand(total_slots, n_k_blocks, dtype=torch.float32, device=device) * 0.1
            B_scale = torch.rand(num_experts, n_k_blocks, n_n_blocks, dtype=torch.float32, device=device) * 0.1
        elif per_channel_quant:
            A_scale = torch.rand(total_slots, 1, dtype=torch.float32, device=device) * 0.1
            B_scale = torch.rand(num_experts, N, dtype=torch.float32, device=device) * 0.1
        else:
            A_scale = torch.rand(total_slots, 1, dtype=torch.float32, device=device) * 0.1
            B_scale = torch.rand(num_experts, 1, dtype=torch.float32, device=device) * 0.1

        # Routing weights
        topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=device)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Build sorted structure
        _, sorted_token_ids, expert_ids, num_tokens_post_padded = (
            _make_sorted_token_ids_and_expert_ids(
                num_tokens, top_k, num_experts, BLOCK_SIZE_M, device
            )
        )

        B_bias = None
        if use_bias:
            B_bias = torch.randn(num_experts, N, dtype=torch.bfloat16, device=device)

        config = {"BLOCK_SIZE_M": BLOCK_SIZE_M, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}

        # Run our implementation
        C_ours = torch.zeros(total_slots, N, dtype=torch.bfloat16, device=device)
        invoke_fused_moe_torch_int8(
            A, B, C_ours, A_scale, B_scale, topk_weights,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            mul_routed_weight, top_k, config, None,
            use_int8_w8a8=True, per_channel_quant=per_channel_quant,
            block_shape=block_shape, B_bias=B_bias,
        )

        # Run naive reference
        C_ref = _naive_fused_moe_int8(
            A, B, A_scale, B_scale, topk_weights,
            sorted_token_ids, expert_ids,
            num_tokens_post_padded.item(), mul_routed_weight, top_k,
            per_channel_quant, block_shape, BLOCK_SIZE_M, B_bias=B_bias,
        )

        # Compare — bf16 has limited precision so use relaxed tolerance
        torch.testing.assert_close(
            C_ours.float(), C_ref.float(), rtol=1e-2, atol=1e-2
        )

    def test_per_tensor_quant(self, device):
        """Test per-tensor quantization (simplest case)."""
        self._run_test(
            device, num_tokens=8, num_experts=4, top_k=2,
            K=64, N=128, per_channel_quant=False,
            block_shape=None, mul_routed_weight=True,
        )

    def test_per_channel_quant(self, device):
        """Test per-channel quantization."""
        self._run_test(
            device, num_tokens=8, num_experts=4, top_k=2,
            K=64, N=128, per_channel_quant=True,
            block_shape=None, mul_routed_weight=True,
        )

    def test_block_quant(self, device):
        """Test block-wise quantization."""
        self._run_test(
            device, num_tokens=8, num_experts=4, top_k=2,
            K=128, N=128, per_channel_quant=False,
            block_shape=[64, 64], mul_routed_weight=True,
        )

    def test_no_routing_weight(self, device):
        """Test without routing weight multiplication."""
        self._run_test(
            device, num_tokens=8, num_experts=4, top_k=2,
            K=64, N=128, per_channel_quant=False,
            block_shape=None, mul_routed_weight=False,
        )

    def test_with_bias(self, device):
        """Test with per-expert bias."""
        self._run_test(
            device, num_tokens=8, num_experts=4, top_k=2,
            K=64, N=128, per_channel_quant=False,
            block_shape=None, mul_routed_weight=True, use_bias=True,
        )

    def test_larger_batch(self, device):
        """Test with more tokens and experts (closer to real scenario)."""
        self._run_test(
            device, num_tokens=32, num_experts=8, top_k=4,
            K=256, N=256, per_channel_quant=True,
            block_shape=None, mul_routed_weight=True,
        )

    def test_single_token(self, device):
        """Edge case: single token."""
        self._run_test(
            device, num_tokens=1, num_experts=4, top_k=2,
            K=64, N=64, per_channel_quant=False,
            block_shape=None, mul_routed_weight=True,
        )
