# SPDX-License-Identifier: Apache-2.0
"""Scoped access to the native PPU ``aten::topk`` implementation.

FlagGems replaces the CUDA dispatch entry globally.  HYV4's lightning indexer
needs the native implementation for its wide 50K row, while MoE routing must
retain FlagGems' top-k tie ordering.  Capture the native kernel before
FlagGems registers its override and invoke the saved handle only from HYV4.
"""

from __future__ import annotations

import torch

_NATIVE_TOPK_KERNEL = None
_EMPTY_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.Undefined)


def capture_native_topk() -> None:
    """Save the current CUDA top-k kernel before FlagGems overrides it."""
    global _NATIVE_TOPK_KERNEL
    if _NATIVE_TOPK_KERNEL is None:
        _NATIVE_TOPK_KERNEL = torch.library.get_kernel("aten::topk", "CUDA")


def native_topk(
    input: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the saved native PPU top-k without changing global dispatch."""
    if _NATIVE_TOPK_KERNEL is None:
        # Unit tests that construct the model outside FLWorker do not run the
        # pre-FlagGems capture hook.  In that environment normal dispatch is
        # the correct fallback.
        return torch.topk(input, k, dim=dim, largest=largest, sorted=sorted)
    return _NATIVE_TOPK_KERNEL.call_boxed(
        _EMPTY_KEYSET,
        input,
        k,
        dim,
        largest,
        sorted,
    )
