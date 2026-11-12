"""Uninstalled Thead stable radix with SSA-resident integer histogram/prefix."""

import torch
import triton
import triton.language as tl
from flag_gems.runtime import device


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


@triton.jit
def _keys(X, Src, row, width, pos, pass_idx):
    valid = pos < width
    saved = tl.load(Src + row * 8192 + pos, valid & (pass_idx > 0), 0, volatile=True)
    logical = tl.where(pass_idx == 0, pos, saved)
    x = tl.load(
        X + row * width + logical, valid & (logical >= 0) & (logical < width), 0
    )
    bits = x.to(tl.uint32, bitcast=True)
    key = tl.where((bits & 0x80000000) != 0, bits, bits ^ 0x7FFFFFFF).to(tl.uint32)
    return logical, bits, key


@triton.jit
def _row_radix(X, P0, P1, Out, width):
    row = tl.program_id(0)
    lane = tl.arange(0, 1024)
    bins = tl.arange(0, 256)
    chunks = tl.arange(0, 8)
    nt = tl.cdiv(width, 1024)
    for pass_idx in range(4):
        Src = tl.where((pass_idx & 1) == 0, P0, P1)
        Dst = tl.where((pass_idx & 1) == 0, P1, P0)
        shift = pass_idx * 8
        hist_all = tl.zeros((8, 256), dtype=tl.int32)
        for chunk in range(nt):
            pos = chunk * 1024 + lane
            logical, bits, key = _keys(X, Src, row, width, pos, pass_idx)
            digit = tl.where(pos < width, (key >> shift) & 255, 0).to(tl.int32)
            hist = tl.histogram(digit, 256)
            pad = 1024 - tl.minimum(1024, width - chunk * 1024)
            hist -= tl.where(bins == 0, pad, 0)
            hist_all += tl.where(chunks[:, None] == chunk, hist[None, :], 0)
        totals = tl.sum(hist_all, axis=0)
        starts = tl.cumsum(totals, axis=0) - totals
        prefix = tl.cumsum(hist_all, axis=0) - hist_all + starts[None, :]
        for chunk in range(nt):
            pos = chunk * 1024 + lane
            logical, bits, key = _keys(X, Src, row, width, pos, pass_idx)
            digit = tl.where(pos < width, (key >> shift) & 255, 256).to(tl.uint32)
            ordered = tl.sort(digit * 1024 + lane.to(tl.uint32), descending=False)
            local = (ordered & 1023).to(tl.int32)
            ordered_digit = (ordered >> 10).to(tl.int32)
            previous = tl.gather(ordered_digit, tl.maximum(lane - 1, 0), axis=0)
            boundary = (lane == 0) | (ordered_digit != previous)
            start = tl.associative_scan(
                tl.where(boundary, lane, 0), axis=0, combine_fn=_maximum
            )
            valid = (chunk * 1024 + local < width) & (ordered_digit < 256)
            # Same integer exclusive prefix; keep its 8x256 matrix in SSA.
            chunk_prefix = tl.sum(tl.where(chunks[:, None] == chunk, prefix, 0), axis=0)
            base = tl.gather(chunk_prefix, tl.minimum(ordered_digit, 255), axis=0)
            rank = base + lane - start
            index = tl.gather(logical, local, axis=0)
            original_bits = tl.gather(bits, local, axis=0)
            valid = valid & (rank >= 0) & (rank < width)
            if pass_idx == 3:
                finite = (original_bits & 0x7F800000) != 0x7F800000
                result = tl.where(finite, index, -1).to(tl.int32)
                tl.store(Out + row * 2048 + rank, result, valid & (rank < 2048))
            else:
                tl.store(Dst + row * 8192 + rank, index.to(tl.int32), valid)
        tl.debug_barrier()


def supports(x, k=2048):
    return (
        device.vendor_name == "thead"
        and k == 2048
        and x.ndim == 2
        and x.dtype == torch.float32
        and x.device.type == "cuda"
        and x.is_contiguous()
        and 0 <= x.shape[0] <= 64
        and 2048 < x.shape[1] <= 8192
    )


def persistent_topk_indices(x, k=2048):
    if not supports(x, k):
        raise NotImplementedError("Thead contiguous F32 M0..64/N2049..8192/k2048 only")
    rows, width = x.shape
    out = torch.empty((rows, 2048), dtype=torch.int32, device=x.device)
    if not rows:
        return out
    p0 = torch.empty((rows, 8192), dtype=torch.int32, device=x.device)
    p1 = torch.empty_like(p0)
    with torch.cuda.device(x.device):
        _row_radix[(rows,)](x, p0, p1, out, width, num_warps=4, num_stages=1)
    return out
