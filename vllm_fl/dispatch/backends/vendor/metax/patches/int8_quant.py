# Copyright (c) 2026 BAAI. All rights reserved.

"""Portable dynamic INT8 quantization for compressed-tensors W8A8 on MetaX.

vLLM's INT8 linear kernels call ``vllm._custom_ops.scaled_int8_quant``.  Its
dynamic (per-token) path dispatches to ``torch.ops._C.dynamic_scaled_int8_quant``,
an operator registered by the NVIDIA CUDA extension.  The MACA torch build does
not register it, so a W8A8 checkpoint fails during
``process_weights_after_loading`` (``get_and_maybe_dequant_weights``) with:

    NotImplementedError: Could not run '_C::dynamic_scaled_int8_quant' with
    arguments from the 'CUDA' backend.

Route the dynamic case to the FlagGems per-token Triton kernel that the
dispatch backend already uses for ``dynamic_per_token_quant_int8``.  The static
per-tensor case has no portable equivalent here and keeps the vendor wrapper.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_APPLIED = False


# The FlagGems Triton kernel must be opaque to ``torch.compile``.  In graph mode
# dynamo traces ``TritonInt8ScaledMMLinearKernel.apply_weights`` and, without
# this wrapper, follows the call all the way into FlagGems' ``libentry.run``,
# where it hits
#
#     for p in self.jit_function.params[len(args):]
#
# and gives up with ``torch._dynamo.exc.Unsupported: Unsupported method call``
# (``Dynamo does not know how to trace method ``__getitem__`` of class
# ``<unknown type>``)``; being raised under fullgraph capture it aborts engine
# startup for every rank (2026-10-26, Hy4-1022 TP32 graph).
#
# A custom op with a fake impl keeps the node in the inductor graph while
# giving AOT/inductor symbolic shapes: dynamo never looks inside, tracing runs
# the fake impl, and the real kernel is launched once at execution time (it is
# already JIT-compiled during ``profile_run``, before CUDA graph capture).
@torch.library.custom_op(
    "vllm_fl_metax::dynamic_per_token_quant_int8", mutates_args=()
)
def _quantize_per_token_int8_op(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm_fl.dispatch.backends.flaggems.impl.quantization import (
        dynamic_per_token_quant_int8_flaggems_triton,
    )

    return dynamic_per_token_quant_int8_flaggems_triton(x)


@_quantize_per_token_int8_op.register_fake
def _(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(x, dtype=torch.int8),
        torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32),
    )


def _dynamic_per_token_int8(
    input: torch.Tensor, symmetric: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not symmetric:
        raise NotImplementedError(
            "MetaX dynamic asymmetric INT8 quantization is not implemented; "
            "compressed-tensors W8A8 checkpoints use the symmetric scheme."
        )
    orig_shape = input.shape
    if input.ndim == 1:
        input = input.reshape(1, -1)
    elif input.ndim > 2:
        input = input.reshape(-1, input.shape[-1])
    quantized, scale = _quantize_per_token_int8_op(input.contiguous())
    return quantized.view(orig_shape), scale, None


def _patch_scaled_int8_quant() -> bool:
    global _APPLIED
    if _APPLIED:
        return True
    try:
        import vllm._custom_ops as vllm_ops
    except Exception as exc:  # noqa: BLE001
        logger.debug("int8-quant: vllm._custom_ops unavailable (%s)", exc)
        return False

    original = getattr(vllm_ops, "scaled_int8_quant", None)
    if original is None or getattr(original, "_fl_metax_int8", False):
        _APPLIED = True
        return original is not None

    def _fl_scaled_int8_quant(input, scale=None, azp=None, symmetric=True):
        if scale is not None:
            # Static per-tensor: no portable replacement, keep the vendor path.
            return original(input, scale, azp, symmetric=symmetric)
        return _dynamic_per_token_int8(input, symmetric)

    _fl_scaled_int8_quant._fl_metax_int8 = True  # type: ignore[attr-defined]
    _fl_scaled_int8_quant._fl_original = original  # type: ignore[attr-defined]
    vllm_ops.scaled_int8_quant = _fl_scaled_int8_quant
    _APPLIED = True
    logger.info(
        "int8-quant: patched vllm._custom_ops.scaled_int8_quant -> FlagGems "
        "Triton per-token INT8 quantization (dynamic path)."
    )
    return True


_patch_scaled_int8_quant()
