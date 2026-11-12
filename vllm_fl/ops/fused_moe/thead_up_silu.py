# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded Thead actual small-M BF16 G1 + SiLU, preserving the original BF16 boundary."""

import flag_gems
import torch
import triton
import triton.language as tl


@triton.jit
def _up_silu_pair(
    A,
    B,
    Out,
    Sorted,
    Experts,
    PostPad,
    R,
    K,
    SA0: tl.constexpr,
    SA1: tl.constexpr,
    SB0: tl.constexpr,
    SB1: tl.constexpr,
    SB2: tl.constexpr,
    SO: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    NAIVE: tl.constexpr,
):
    pid = tl.program_id(0)
    group = pid // (H // BN)
    column = pid % (H // BN)
    padded = tl.load(PostPad)
    if group * BM >= padded:
        return
    position = group * BM + tl.arange(0, BM).to(tl.int64)
    if NAIVE:
        token = tl.where(tl.arange(0, BM) == 0, group, R).to(tl.int64)
    else:
        token = tl.load(Sorted + position).to(tl.int64)
    valid = token < R
    expert = tl.load(Experts + group).to(tl.int64)
    n = column * BN + tl.arange(0, BN).to(tl.int64)
    output = Out + token[:, None] * SO + n[None, :]
    if expert < 0:
        tl.store(output, 0.0, mask=valid[:, None])
        return
    kk = tl.arange(0, BK)
    ap = A + (token[:, None] // TOPK) * SA0 + kk[None, :] * SA1
    bg = B + expert * SB0 + kk[:, None] * SB2 + n[None, :] * SB1
    bu = B + expert * SB0 + kk[:, None] * SB2 + (n[None, :] + H) * SB1
    gate = tl.zeros((BM, BN), dtype=tl.float32)
    up = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BK)):
        a = tl.load(ap, mask=valid[:, None], other=0.0)
        b_gate = tl.load(bg)
        b_up = tl.load(bu)
        gate += tl.dot(a, b_gate)
        up += tl.dot(a, b_up)
        ap += BK * SA1
        bg += BK * SB2
        bu += BK * SB2
    # Original G1 stores BF16 before the separately dispatched SiLU. Keep it.
    gate_rounded = gate.to(tl.bfloat16).to(tl.float32)
    up_rounded = up.to(tl.bfloat16)
    activated = tl.fdiv(gate_rounded, 1.0 + tl.exp(-gate_rounded)) * up_rounded
    tl.store(output, activated.to(tl.bfloat16), mask=valid[:, None])


def _span(x):
    start = x.storage_offset() * x.element_size()
    return start, start + x.numel() * x.element_size()


def _eligible(a, b, out, sorted_ids, expert_ids, post_pad):
    """Actual BM16/S4 small-M naive or sorted routes; no quant/bias/LoRA/clamp/router multiply.

    Production caller must separately guard activation=SiLU and all unsupported
    MoE semantics. This private primitive rejects unsupported metadata only;
    producer guarantees token IDs in[0,M*8] and expert IDs in[-1,E).
    """
    if (
        not all(isinstance(x, torch.Tensor) for x in (a, b, out, expert_ids, post_pad))
        or sorted_ids is not None
    ):
        return False
    if flag_gems.vendor_name != "thead":
        return False
    if not (a.ndim == 2 and b.ndim == 3 and out.ndim == 2):
        return False
    m, k = a.shape
    if not (
        1 <= m <= 8
        and k == 4096
        and b.shape[0] > 0
        and b.shape[1:] == (512, 4096)
        and out.shape == (m * 8, 256)
    ):
        return False
    naive = m <= 8
    if naive != (sorted_ids is None):
        return False
    route_tensors = (
        (expert_ids, post_pad) if naive else (sorted_ids, expert_ids, post_pad)
    )
    tensors = (a, b, out, *route_tensors)
    if a.device.type != "cuda" or any(x.device != a.device for x in tensors):
        return False
    if any(x.dtype != torch.bfloat16 for x in (a, b, out)):
        return False
    if any(
        x.dtype != torch.int32 or x.ndim != 1 or not x.is_contiguous()
        for x in route_tensors
    ):
        return False
    if not all(x.is_contiguous() for x in (a, b, out)) or post_pad.numel() != 1:
        return False
    r = m * 8
    if naive:
        em = r * 16
        if expert_ids.numel() != r:
            return False
    else:
        em = sorted_ids.numel()
        if em < r or em % 16 or expert_ids.numel() < triton.cdiv(em, 16):
            return False
        if m < 16:
            em = min(em, r * 16)
    for x in (a, b, *route_tensors):
        if x.untyped_storage().data_ptr() == out.untyped_storage().data_ptr():
            lo, hi = _span(x)
            ol, oh = _span(out)
            if lo < oh and ol < hi:
                return False
    return True


def _launch(a, b, out, sorted_ids, expert_ids, post_pad):
    m, k = a.shape
    r = m * 8
    em = r * 16
    naive = True
    _up_silu_pair[(triton.cdiv(em, 16) * 4,)](
        a,
        b,
        out,
        sorted_ids,
        expert_ids,
        post_pad,
        r,
        k,
        *a.stride(),
        *b.stride(),
        out.stride(0),
        16,
        64,
        128,
        8,
        256,
        naive,
        num_warps=4,
        num_stages=4,
    )
    return out


def up_silu_out(a, b, out, sorted_ids, expert_ids, post_pad):
    if not _eligible(a, b, out, sorted_ids, expert_ids, post_pad):
        raise ValueError("Unsupported bounded Thead M1..8 naive BF16 G1+SiLU metadata")
    return _launch(a, b, out, sorted_ids, expert_ids, post_pad)


def try_up_silu_out(a, b, out, sorted_ids, expert_ids, post_pad):
    """Metadata-only fallback; errors from an eligible kernel propagate."""
    if not _eligible(a, b, out, sorted_ids, expert_ids, post_pad):
        return False
    _launch(a, b, out, sorted_ids, expert_ids, post_pad)
    return True
