"""Thead task-local M1..64 integer COMPACT_TO_FRONT; not installed."""

import torch
import triton
import triton.language as tl
from flag_gems.runtime import device


@triton.jit
def _physical_compact(
    Req,
    Table,
    Tokens,
    Out,
    Lengths,
    RS: tl.constexpr,
    TS0: tl.constexpr,
    TS1: tl.constexpr,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    row = tl.program_id(0)
    pos = tl.arange(0, 2048)
    req = tl.load(Req + row * RS).to(tl.int64)
    req = tl.where(req < 0, req + ROWS, req)
    token = tl.load(Tokens + row * IS0 + pos * IS1).to(tl.int64)
    logical = tl.maximum(token, 0) // 64
    valid = (token >= 0) & (logical < COLS)
    block = tl.load(Table + req * TS0 + logical * TS1, mask=valid, other=-1).to(
        tl.int64
    )
    valid = valid & (block >= 0)
    physical = block * 64 + tl.maximum(token, 0) % 64
    count = tl.sum(valid.to(tl.int32), 0)
    # Fast path is a tensor-value branch, not a request/dataset distinction.
    if count == 2048:
        tl.store(Out + row * 2048 + pos, physical.to(tl.int32))
    else:
        rank = tl.cumsum(valid.to(tl.int32), 0) - 1
        # Valid destinations [0,count) and padding [count,2048) are disjoint.
        # Every output is written exactly once; no barriers/atomics/sink race.
        tl.store(Out + row * 2048 + pos, -1, mask=pos >= count)
        tl.store(Out + row * 2048 + rank, physical.to(tl.int32), mask=valid)
    tl.store(Lengths + row, count)


def supports(request_ids, block_table, token_indices, block_size):
    ints = (torch.int32, torch.int64)
    return (
        device.vendor_name == "thead"
        and block_size == 64
        and request_ids.ndim == 1
        and block_table.ndim == 2
        and token_indices.ndim == 2
        and request_ids.dtype in ints
        and block_table.dtype in ints
        and token_indices.dtype in ints
        and request_ids.device.type == "cuda"
        and request_ids.device == block_table.device == token_indices.device
        and 1 <= token_indices.shape[0] <= 4096
        and token_indices.shape[1] == 2048
        and request_ids.numel() == token_indices.shape[0]
        and block_table.shape[0] > 0
        and 1 <= block_table.shape[1] <= 1024
        and all(
            s > 0 for x in (request_ids, block_table, token_indices) for s in x.stride()
        )
    )


def _convert_supported(request_ids, block_table, token_indices, block_size):
    """Private: caller must have validated supports(...) True; data preconditions unchanged."""
    n = token_indices.shape[0]
    out = torch.empty((n, 2048), device=token_indices.device, dtype=torch.int32)
    lengths = torch.empty((n,), device=token_indices.device, dtype=torch.int32)
    with torch.cuda.device(token_indices.device):
        _physical_compact[n,](
            request_ids,
            block_table,
            token_indices,
            out,
            lengths,
            request_ids.stride(0),
            *block_table.stride(),
            *token_indices.stride(),
            block_table.shape[0],
            block_table.shape[1],
            num_warps=4,
            num_stages=1,
        )
    return (out, lengths)


def convert_request_to_physical_indices(
    request_ids, block_table, token_indices, block_size
):
    """Exact signed-integer output/length; valid IDs in [-table_rows,table_rows).
    Same request-value precondition as PyTorch advanced indexing. No values are
    copied to CPU to enforce it. Unsupported metadata must use caller's old path.
    """
    if not supports(request_ids, block_table, token_indices, block_size):
        raise NotImplementedError(
            "Thead M1..64/K2048/block64/signed32or64/positive strides only"
        )
    return _convert_supported(request_ids, block_table, token_indices, block_size)
