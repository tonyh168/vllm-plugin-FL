# SPDX-License-Identifier: Apache-2.0
"""Raw-byte lookup used by Flash Engram on CUDA-alike devices."""
from vllm.triton_utils import tl, triton

@triton.jit
def _lookup_ple_embedding_from_pinned_kernel(
    weight_ptr: tl.pointer_type(tl.uint8),  # type: ignore[valid-type]
    ids_ptr,
    output_ptr: tl.pointer_type(tl.uint8),  # type: ignore[valid-type]
    row_bytes,
    tp_vocab_start,
    tp_vocab_end,
    BLOCK_D: tl.constexpr,
):
    """Copy TP-owned PLE rows as raw bytes from a CUDA view of pinned host memory.

    Byte pointers keep the storage dtype out of the kernel signature, so
    dtypes Triton cannot lower on the GPU (FP8 E4M3FN before SM89) still work.
    """
    row_id = tl.program_id(0).to(tl.int64)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    store_mask = offsets < row_bytes
    load_mask = store_mask & in_range
    values = tl.load(
        weight_ptr + local_idx * row_bytes + offsets,
        mask=load_mask,
        other=0,
    )
    tl.store(
        output_ptr + row_id * row_bytes + offsets,
        values,
        mask=store_mask,
    )

