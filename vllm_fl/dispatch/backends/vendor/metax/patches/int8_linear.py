# Copyright (c) 2026 BAAI. All rights reserved.

"""Keep the INT8 W8A8 Triton scaled-mm kernel opaque to ``torch.compile``.

MACA has no ``cutlass_scaled_mm``, so ``compressed-tensors`` W8A8 dense linear
falls back to vLLM's Triton kernel (``TritonInt8ScaledMMLinearKernel``), whose
``triton_scaled_mm`` picks tile shapes from the token count::

    next_power_of_2_M = max(32, triton.next_power_of_2(M))

``triton.next_power_of_2`` needs a concrete integer, so under ``torch.compile``
dynamo *specializes* the token dimension.  vLLM marks that dimension dynamic
(``dynamic_arg_dims``), and the two conflict while the decode graph's guards are
being built::

    ConstraintViolationError: Constraints violated (L['inputs_embeds'].size()[0])!
      - You marked L['inputs_embeds'].size()[0] as dynamic but your code
        specialized it to be a constant (8192).

Engine startup aborts on every rank (2026-10-26, Hy4-1022 TP32 graph).  On CUDA
the cutlass path is taken instead, which is why this only shows up here.

Wrapping the kernel in a custom op hides the heuristic from dynamo: the fake
impl gives symbolic shapes, the real kernel (and its tile selection) only runs
at execution time.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_APPLIED = False


def _install() -> bool:
    global _APPLIED
    if _APPLIED:
        return True
    try:
        import vllm.model_executor.kernels.linear.scaled_mm.triton as kernel_mod
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
            triton_scaled_mm as original,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("int8-linear: scaled-mm module unavailable (%s)", exc)
        return False

    if getattr(kernel_mod.triton_scaled_mm, "_fl_metax_opaque", False):
        _APPLIED = True
        return True

    @torch.library.custom_op("vllm_fl_metax::triton_scaled_mm", mutates_args=())
    def _triton_scaled_mm_op(
        input: torch.Tensor,
        weight: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        return original(
            input,
            weight,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=out_dtype,
            bias=bias,
        )

    @_triton_scaled_mm_op.register_fake
    def _(
        input: torch.Tensor,
        weight: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.empty(
            (input.shape[0], weight.shape[1]),
            dtype=out_dtype,
            device=input.device,
        )

    def _opaque_triton_scaled_mm(
        input,
        weight,
        scale_a,
        scale_b,
        out_dtype,
        bias=None,
        **_kwargs,
    ):
        return _triton_scaled_mm_op(
            input, weight, scale_a, scale_b, bias, out_dtype
        )

    _opaque_triton_scaled_mm._fl_metax_opaque = True  # type: ignore[attr-defined]
    _opaque_triton_scaled_mm._fl_original = original  # type: ignore[attr-defined]
    kernel_mod.triton_scaled_mm = _opaque_triton_scaled_mm
    _APPLIED = True
    logger.info(
        "int8-linear: triton_scaled_mm wrapped in a custom op so the tile "
        "heuristic cannot specialize the dynamic token dimension."
    )
    return True


_install()
