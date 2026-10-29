# SPDX-License-Identifier: Apache-2.0
"""FlagGems FP8 (block-wise 128x128) MoE experts for out-of-tree platforms.

vLLM >= 0.24 only routes FP8 MoE to backends that support the *block-wise*
quant key used by DeepSeek/GLM checkpoints
(``kFp8Static128BlockSym x dynamic(1,128)``).  None of the in-tree kernels are
available on T-Head PPU, so the oracle returns ``Fp8MoeBackend.NONE`` and
``Fp8LinearMethod.process_weights_after_loading`` raises::

    ValueError: Unsupported FP8 MoE backend: NONE

This module mirrors the existing INT8 W8A8 adapter
(``vllm_fl.quantization.w8a8.moe``): it keeps the activations in floating
point (``expects_unquantized_inputs``) and lets FlagGems
``fused_experts_impl`` own routing, activation quant, GEMM and the final
dequant, using ``block_shape`` = the checkpoint's ``weight_block_size``.
"""

from __future__ import annotations

import torch

from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts


def _flaggems_fused_experts(**kwargs) -> torch.Tensor:
    import flag_gems

    return flag_gems.fused_experts_impl(**kwargs)


class TritonFp8ExpertsFL(TritonExperts):
    """FP8 W8A8 (block-wise) MoE executed by FlagGems ``fused_experts_impl``.

    The checkpoint contract is ``w1=[E, 2I, K]`` / ``w2=[E, K, I]`` in FP8 with
    per-block scales of shape ``[E, 2I/128, K/128]`` and ``[E, K/128, I/128]``;
    FlagGems produces the dynamic per-token-group activation scales internally.
    """

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        # PPU reports through the FL platform; the in-tree check only allows
        # CUDA-alike/XPU via torch's canned list.
        return True

    @classmethod
    def is_supported_config(
        cls,
        moe_config,
        weight_key,
        activation_key,
        activation_format,
    ) -> tuple[bool, str | None]:
        """Accept the block-wise FP8 scheme the in-tree Triton kernel rejects.

        Everything is inherited from ``FusedMoEExperts`` except the device gate
        (the FlagGems path runs wherever ``flag_gems.fused_experts_impl`` does)
        and the quant-scheme gate (``kFp8Static128BlockSym`` is exactly what we
        implement here).
        """
        from vllm.model_executor.layers.fused_moe.modular_kernel import (
            FusedMoEActivationFormat,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8Static128BlockSym,
        )

        if weight_key != kFp8Static128BlockSym:
            return False, f"kernel does not support quantization scheme {weight_key}"
        if activation_format != FusedMoEActivationFormat.Standard:
            return False, f"{activation_format.value} activation format"
        if not (moe_config.is_act_and_mul or cls._supports_no_act_and_mul()):
            return False, "no act_and_mul MLP layer"
        if not cls._supports_activation(moe_config.activation):
            return False, f"{moe_config.activation} activation"
        if not cls._supports_parallel_config(moe_config.moe_parallel_config):
            return False, f"parallel config {moe_config.moe_parallel_config}"
        if not cls._supports_routing_method(
            moe_config.routing_method, weight_key, activation_key
        ):
            return False, f"routing method {moe_config.routing_method}"
        if not cls._supports_shape(moe_config.hidden_dim):
            return False, f"{moe_config.hidden_dim} hidden dim is not supported"
        return True, None

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        del workspace13, workspace2, expert_tokens_meta

        quant_config = self.quant_config
        if not quant_config.use_fp8_w8a8:
            raise ValueError("TritonFp8ExpertsFL requires an FP8 W8A8 quant config")
        if getattr(self, "_lora_context", None) is not None:
            raise NotImplementedError(
                "The FlagGems FP8 MoE adapter does not support LoRA"
            )
        if a1q_scale is not None or a2_scale is not None:
            raise ValueError(
                "FP8 activation was quantized before the FlagGems experts "
                "adapter; expected an unquantized floating-point input"
            )
        if hidden_states.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError(
                "FP8 functional experts require floating-point hidden states, "
                f"got {hidden_states.dtype}"
            )
        if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
            raise TypeError("TritonFp8ExpertsFL requires fp8_e4m3fn expert weights")
        if quant_config.gemm1_clamp_limit is not None:
            # FlagGems fused_experts_impl has no clamped-activation knob.
            raise NotImplementedError(
                "FlagGems FP8 MoE does not support gemm1_clamp_limit"
            )

        result = _flaggems_fused_experts(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation=activation.value,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_fp8_w8a8=True,
            per_channel_quant=quant_config.per_act_token_quant,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=quant_config.w1_scale,
            w2_scale=quant_config.w2_scale,
            a1_scale=quant_config.a1_scale,
            a2_scale=quant_config.a2_scale,
            block_shape=quant_config.block_shape,
            w1_bias=quant_config.w1_bias,
            w2_bias=quant_config.w2_bias,
        )
        output.copy_(result)


__all__ = ["TritonFp8ExpertsFL"]
