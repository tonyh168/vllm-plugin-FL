# SPDX-License-Identifier: Apache-2.0
"""FP8 MoE oracle adapter for out-of-tree (OOT) platforms such as T-Head PPU.

``vllm.model_executor.layers.fused_moe.oracle.fp8.select_fp8_moe_backend``
returns ``Fp8MoeBackend.NONE`` on OOT platforms, which makes
``Fp8LinearMethod.process_weights_after_loading`` raise::

    ValueError: Unsupported FP8 MoE backend: NONE

This module mirrors ``vllm_fl.quantization.w8a8.moe`` (the INT8 equivalent):
when the platform is out of tree, OOT registration is enabled and the
checkpoint uses the standard block-wise FP8 scheme
(``kFp8Static128BlockSym`` x dynamic ``(1, 128)`` — DeepSeek/GLM style), it
returns the FlagGems-backed ``TritonFp8ExpertsFL`` experts implementation.
Anything else falls through to the untouched upstream selector.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_ADAPTER_MARKER = "_vllm_fl_fp8_moe_adapter"


def install_fl_fp8_moe_selector() -> bool:
    """Patch the upstream FP8 MoE backend oracle. Idempotent.

    Returns True when the adapter is installed (or was already installed).
    """
    try:
        from vllm.model_executor.layers.fused_moe.oracle import fp8 as oracle_module
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8Static128BlockSym,
        )
    except Exception as exc:  # pragma: no cover - version drift guard
        logger.debug("FP8 MoE oracle unavailable, skipping adapter: %s", exc)
        return False

    current_selector = oracle_module.select_fp8_moe_backend
    if getattr(current_selector, _ADAPTER_MARKER, False):
        return True

    def select_fp8_moe_backend_fl(
        config,
        weight_key=None,
        activation_key=None,
        allow_vllm_cutlass: bool = False,
    ):
        from vllm.platforms import current_platform

        from vllm_fl.utils import is_oot_enabled

        block_fp8 = weight_key is not None and weight_key == kFp8Static128BlockSym
        use_fl = (
            block_fp8
            and current_platform.is_out_of_tree()
            and is_oot_enabled()
            and not getattr(config, "is_lora_enabled", False)
            and not config.moe_parallel_config.use_batched_activation_format
        )
        if not use_fl:
            return current_selector(
                config,
                weight_key,
                activation_key,
                allow_vllm_cutlass,
            )

        from vllm_fl.quantization.fp8_moe.moe_experts import TritonFp8ExpertsFL

        logger.info_once(
            "Using TRITON FP8 MoE backend (FlagGems experts) for block-wise FP8."
        )
        return oracle_module.Fp8MoeBackend.TRITON, TritonFp8ExpertsFL

    setattr(select_fp8_moe_backend_fl, _ADAPTER_MARKER, True)
    oracle_module.select_fp8_moe_backend = select_fp8_moe_backend_fl

    # vllm.model_executor.layers.quantization.fp8 imported the selector by name.
    try:
        from vllm.model_executor.layers.quantization import fp8 as scheme_module

        scheme_module.select_fp8_moe_backend = select_fp8_moe_backend_fl
    except Exception as exc:  # pragma: no cover
        logger.debug("Could not patch quantization.fp8 selector: %s", exc)

    return True


__all__ = ["install_fl_fp8_moe_selector"]
