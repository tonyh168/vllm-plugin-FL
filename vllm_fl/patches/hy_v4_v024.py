# Copyright (c) 2026 BAAI. All rights reserved.
"""Install HY4 support into a pristine vLLM 0.24 runtime.

All changes stay inside vllm-plugin-FL. The hook registers the checkpoint
config, compressed-MLA architecture conversion, lazy model implementation,
and expert-sliced safetensors loader without modifying the vLLM installation.
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any

from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)

from vllm_fl.configs.hy_v4 import HYV4Config
from vllm_fl.model_loader.hy_v4_loader import HYV4SafetensorsLoader
from vllm_fl.patches._version import is_vllm_024

logger = logging.getLogger(__name__)

_ARCHITECTURE = "HYV4ForCausalLM"
_LOAD_FORMAT = "hy4_safetensors"


def _patch_mxfp8_override_order(me_quant: Any) -> None:
    """Make vLLM 0.24 probe the canonical ModelOpt MXFP8 entry first.

    vLLM 0.24 maps both ``modelopt_mxfp8`` and the online shorthand
    ``mxfp8`` to ``ModelOptMxFp8Config``, but only the former is present in
    ``ModelConfig._verify_quantization``'s ordered override list.  Therefore
    the shorthand reports an override before the canonical entry is reached
    and ModelConfig rejects a serialized MXFP8 checkpoint.  Returning a
    no-override view for the shorthand is equivalent to placing ``mxfp8``
    after ``modelopt_mxfp8`` in that list, without replacing ModelConfig or
    modifying the vLLM installation.
    """
    current_getter = me_quant.get_quantization_config
    if getattr(current_getter, "_hy4_v024_mxfp8_order", False):
        return

    mxfp8_config = current_getter("mxfp8")

    class MXFP8AliasAfterCanonical(mxfp8_config):
        @classmethod
        def override_quantization_method(
            cls,
            hf_quant_cfg: dict[str, Any],
            user_quant: str | None,
            hf_config: Any = None,
        ) -> None:
            return None

    @wraps(current_getter)
    def get_quantization_config(name: str):
        if name == "mxfp8":
            return MXFP8AliasAfterCanonical
        return current_getter(name)

    get_quantization_config._hy4_v024_mxfp8_order = True
    me_quant.get_quantization_config = get_quantization_config


class HYV4ModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Expose HY4's compressed MLA and ModelOpt MXFP8 metadata to vLLM."""

    def get_head_size(self) -> int:
        return int(self.hf_text_config.kv_lora_rank) + int(
            self.hf_text_config.qk_rope_head_dim
        )

    def is_deepseek_mla(self) -> bool:
        return True

    def get_quantization_config(self) -> dict[str, Any] | None:
        quant_config = super().get_quantization_config()
        if quant_config is None or quant_config.get("quant_method") != "mxfp8":
            return quant_config

        # vLLM 0.24's ModelOpt MXFP8 parser already understands the raw
        # MiniMax-style schema during weight construction, but its earlier
        # override-selection pass only recognizes a ModelOpt-shaped config.
        # Return a normalized copy for architecture detection; keep the HF
        # config untouched so ModelOptMxFp8Config.from_config handles it later.
        return {
            "quant_method": "modelopt",
            "quantization": {
                "quant_algo": "MXFP8",
                "kv_cache_quant_algo": quant_config.get("kv_cache_quant_algo"),
                "exclude_modules": quant_config.get("ignored_layers", []) or [],
            },
        }


def _patch_mla_prefill_for_non_cuda() -> None:
    """Patch MLA prefill backend availability for non-CUDA platforms.

    vLLM's FlashAttnPrefillBackend.is_available() delegates to
    is_flash_attn_varlen_func_available() which only returns True for CUDA/XPU.
    MetaX has its own flash_attn library with flash_attn_varlen_func, so we
    patch the check to recognize it.
    """
    from vllm.platforms import current_platform

    if current_platform.is_cuda() or current_platform.is_xpu():
        return

    # Check if flash_attn is actually available on this platform
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
        has_flash_attn = True
    except ImportError:
        has_flash_attn = False

    if not has_flash_attn:
        logger.warning(
            "flash_attn not available on this platform, "
            "MLA prefill backend will use torch SDPA fallback"
        )
        _patch_mla_prefill_torch_fallback()
        return

    # Patch is_flash_attn_varlen_func_available to return True
    try:
        from vllm.v1.attention.backends import fa_utils
        fa_utils.is_flash_attn_varlen_func_available = lambda: True

        # Also need to ensure flash_attn_varlen_func is loaded in the module
        from vllm.v1.attention.backends.mla.prefill import flash_attn as mla_fa
        if mla_fa.flash_attn_varlen_func is None:
            from flash_attn import flash_attn_varlen_func as _fa_varlen
            mla_fa.flash_attn_varlen_func = _fa_varlen

        logger.info(
            "Patched is_flash_attn_varlen_func_available() for MetaX platform"
        )
    except Exception as e:
        logger.warning(f"Failed to patch flash_attn availability: {e}, "
                       "falling back to torch SDPA")
        _patch_mla_prefill_torch_fallback()


def _patch_mla_prefill_torch_fallback() -> None:
    """Register a torch SDPA fallback MLA prefill backend."""
    try:
        from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
        from vllm.v1.attention.backends.mla.prefill import selector as mla_selector
    except ImportError:
        logger.debug("MLA prefill selector not available, skipping patch")
        return

    class TorchSDPAMLAPrefillBackend(MLAPrefillBackend):
        """Torch SDPA fallback MLA prefill backend for non-CUDA platforms."""

        @staticmethod
        def get_name() -> str:
            return "TORCH_SDPA_MLA"

        @classmethod
        def is_available(cls) -> bool:
            return True

        @classmethod
        def supports_compute_capability(cls, device_capability) -> bool:
            return True

        @classmethod
        def validate_configuration(cls, device_capability, selector_config):
            return []

        def __init__(self, num_heads, scale, kv_lora_rank,
                     qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
                     vllm_config) -> None:
            super().__init__(
                num_heads=num_heads, scale=scale,
                kv_lora_rank=kv_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim, vllm_config=vllm_config,
            )

        def run_prefill_new_tokens(self, q, k, v, return_softmax_lse,
                                   out=None, output_scale=None):
            import torch
            import torch.nn.functional as F
            q_t = q.transpose(0, 1)
            k_t = k.transpose(0, 1)
            v_padded = F.pad(v, [0, q.shape[-1] - v.shape[-1]]) if v.shape[-1] < q.shape[-1] else v
            v_t = v_padded.transpose(0, 1)
            attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=self.scale, is_causal=True)
            attn_out = attn_out.transpose(0, 1)[..., :v.shape[-1]]
            if return_softmax_lse:
                scores = torch.einsum('thd,shd->ths', q_t, k_t) * self.scale
                lse = torch.logsumexp(scores, dim=-1)
                return attn_out, lse
            return attn_out

        def run_prefill_context_chunk(self, chunk_idx, q, k, v):
            import torch
            import torch.nn.functional as F
            q_t = q.transpose(0, 1)
            k_t = k.transpose(0, 1)
            v_padded = F.pad(v, [0, q.shape[-1] - v.shape[-1]]) if v.shape[-1] < q.shape[-1] else v
            v_t = v_padded.transpose(0, 1)
            attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=self.scale, is_causal=False)
            attn_out = attn_out.transpose(0, 1)[..., :v.shape[-1]]
            scores = torch.einsum('thd,shd->ths', q_t, k_t) * self.scale
            lse = torch.logsumexp(scores, dim=-1)
            return attn_out, lse

    def _patched_get_mla_prefill_backend(vllm_config):
        return TorchSDPAMLAPrefillBackend

    mla_selector.get_mla_prefill_backend = _patched_get_mla_prefill_backend
    try:
        from vllm.model_executor.layers.attention import mla_attention
        mla_attention.get_mla_prefill_backend = _patched_get_mla_prefill_backend
    except (ImportError, AttributeError):
        pass
    logger.info("Registered torch SDPA MLA prefill fallback backend")


def _patch_int8_moe_for_metax() -> None:
    """Patch TritonExperts._supports_quant_scheme for MetaX INT8 support.

    vLLM checks `is_cuda()` (nvidia-only) for INT8 MoE support.
    MetaX hardware supports INT8 via MACA but reports as cuda_alike,
    not nvidia. We patch the check to include cuda_alike platforms.
    """
    from vllm.platforms import current_platform

    if current_platform.is_cuda():
        return

    if not current_platform.is_cuda_alike():
        return

    try:
        from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
            TritonExperts,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            QuantKey,
            kInt8DynamicTokenSym,
            kInt8StaticChannelSym,
        )

        _orig_supports_quant_scheme = TritonExperts._supports_quant_scheme.__func__

        @staticmethod
        def _patched_supports_quant_scheme(
            weight_key: "QuantKey | None",
            activation_key: "QuantKey | None",
        ) -> bool:
            from vllm.model_executor.layers.quantization.utils.quant_utils import (
                kFp8Dynamic128Sym,
                kFp8DynamicTokenSym,
                kFp8DynamicTensorSym,
                kFp8Static128BlockSym,
                kFp8StaticChannelSym,
                kFp8StaticTensorSym,
            )

            # MetaX supports INT8 (similar to Turing+)
            supported: list[tuple["QuantKey | None", "QuantKey | None"]] = [
                (None, None),
                (kInt8StaticChannelSym, kInt8DynamicTokenSym),
            ]
            if current_platform.supports_fp8():
                supported += [
                    (kFp8Static128BlockSym, kFp8Dynamic128Sym),
                    (kFp8StaticChannelSym, kFp8DynamicTokenSym),
                    (kFp8StaticTensorSym, kFp8DynamicTokenSym),
                    (kFp8StaticTensorSym, kFp8StaticTensorSym),
                    (kFp8StaticTensorSym, kFp8DynamicTensorSym),
                ]
            return (weight_key, activation_key) in supported

        TritonExperts._supports_quant_scheme = _patched_supports_quant_scheme
        logger.info("Patched TritonExperts INT8 MoE support for MetaX platform")
    except Exception as e:
        logger.warning(f"Failed to patch INT8 MoE support: {e}")


def apply_hy_v4_v024_patches() -> bool:
    """Register the HY4 runtime components required by vLLM 0.24.x."""
    if not is_vllm_024():
        return False

    from vllm.model_executor import model_loader
    from vllm.model_executor.layers import quantization as me_quant
    from vllm.model_executor.models import registry as model_registry
    from vllm.transformers_utils import (
        config as transformers_config,
        model_arch_config_convertor,
    )

    transformers_config._CONFIG_REGISTRY.setdefault("hy_v4", HYV4Config)
    model_arch_config_convertor.MODEL_ARCH_CONFIG_CONVERTORS["hy_v4"] = (
        HYV4ModelArchConfigConvertor
    )
    model_registry.ModelRegistry.register_model(
        _ARCHITECTURE,
        "vllm_fl.models.hy_v4:HYV4ForCausalLM",
    )
    _patch_mxfp8_override_order(me_quant)

    registered_loaders = model_loader._LOAD_FORMAT_TO_MODEL_LOADER
    if registered_loaders.get(_LOAD_FORMAT) is not HYV4SafetensorsLoader:
        model_loader.register_model_loader(_LOAD_FORMAT)(HYV4SafetensorsLoader)

    # Patch MLA prefill backend for non-CUDA platforms (MetaX)
    _patch_mla_prefill_for_non_cuda()

    # Patch INT8 MoE support for MetaX (cuda_alike but not nvidia)
    _patch_int8_moe_for_metax()

    logger.info("Installed HY4 runtime compatibility for vLLM 0.24")
    return True


__all__ = [
    "HYV4ModelArchConfigConvertor",
    "apply_hy_v4_v024_patches",
]
