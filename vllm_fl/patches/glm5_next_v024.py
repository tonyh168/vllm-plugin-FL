# SPDX-License-Identifier: Apache-2.0
"""Register the plugin-owned GLM5-Next implementation on vLLM 0.24."""

import logging
import os
from functools import wraps
from types import ModuleType

import torch

from vllm.model_executor.models.config import (
    HybridAttentionMambaModelConfig,
)
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)

from vllm_fl.patches._version import is_vllm_024
from vllm_fl.kernels.glm5_next.provider import (
    get_glm5_provider,
    use_nvidia_reference,
)

logger = logging.getLogger(__name__)

_CAUSAL_ARCH = "Glm5NextForCausalLM"
_CONDITIONAL_ARCH = "Glm5NextForConditionalGeneration"
_MHC_CUDA_MAX_TOKENS = 0


def _is_missing_cache_op(exc: AttributeError, op_name: str) -> bool:
    """Return whether vLLM failed because ``_C_cache_ops`` lacks an op."""
    message = str(exc)
    return "_C_cache_ops" in message and op_name in message


def _has_vllm_cache_op(op_name: str) -> bool:
    """Probe the extension ABI without invoking a device kernel."""
    try:
        getattr(torch.ops._C_cache_ops, op_name)
    except AttributeError:
        return False
    return True


def _concat_and_cache_mla_bf16_fallback(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    """Correctness fallback for a vendor MLA backend without its cache op.

    This is deliberately limited to BF16 cache semantics.  Quantized cache
    formats require the vendor or FlagGems implementation because ``scale``
    participates in the stored representation.
    """
    del scale
    if kv_cache_dtype not in ("auto", "bfloat16"):
        raise NotImplementedError(
            "GLM5-Next portable concat_and_cache_mla only supports BF16 KV "
            f"cache, got {kv_cache_dtype!r}"
        )
    source = (
        kv_c
        if k_pe.shape[-1] == 0
        else torch.cat((kv_c, k_pe), dim=-1)
    )
    slots = slot_mapping.flatten().to(torch.int64)
    valid = slots >= 0
    cache_flat = kv_cache.view(-1, kv_cache.shape[-1])
    cache_flat[slots[valid]] = source[valid]


def _install_mla_boundary_compat_ops(
    custom_ops: ModuleType | None = None,
) -> bool:
    """Keep vendor sparse MLA while filling its optional vLLM ABI edges.

    Vendor backends remain responsible for the actual sparse attention.  The
    wrappers below only cover GLM5-Next's zero-width RoPE query and the common
    BF16 cache-write ABI that some OOT vLLM builds do not provide.
    """
    if custom_ops is None:
        from vllm import _custom_ops as custom_ops

    installed = False
    strict_flaggems = get_glm5_provider() == "flaggems"

    concat_mla_q = custom_ops.concat_mla_q
    if not getattr(concat_mla_q, "_glm5_next_nope_fix", False):

        @wraps(concat_mla_q)
        def concat_mla_q_nope(ql_nope, q_pe, q_out):
            if strict_flaggems:
                raise RuntimeError(
                    "VLLM_FL_GLM5_PROVIDER=flaggems was requested, but the "
                    "stock/vendor sparse-MLA path called "
                    "vllm._custom_ops.concat_mla_q. The worker did not select "
                    "FlagGemsSparseMLABackend; check the worker environment, "
                    "Plugin FL patch/version, and vendor backend overrides."
                )
            if q_pe.shape[-1] == 0:
                q_out.copy_(ql_nope)
                return None
            return concat_mla_q(ql_nope, q_pe, q_out)

        concat_mla_q_nope._glm5_next_nope_fix = True
        concat_mla_q_nope._glm5_next_original = concat_mla_q
        custom_ops.concat_mla_q = concat_mla_q_nope
        installed = True

    concat_and_cache_mla = custom_ops.concat_and_cache_mla
    if not _has_vllm_cache_op("concat_and_cache_mla") and not getattr(
        concat_and_cache_mla, "_glm5_next_vendor_fallback", False
    ):
        native_cache_op_available = True
        flaggems_cache_writer = None
        flaggems_cache_writer_loaded = False

        @wraps(concat_and_cache_mla)
        def concat_and_cache_mla_vendor_first(
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            kv_cache_dtype,
            scale,
        ):
            nonlocal native_cache_op_available
            nonlocal flaggems_cache_writer
            nonlocal flaggems_cache_writer_loaded

            if native_cache_op_available:
                try:
                    return concat_and_cache_mla(
                        kv_c,
                        k_pe,
                        kv_cache,
                        slot_mapping,
                        kv_cache_dtype,
                        scale,
                    )
                except AttributeError as exc:
                    if not _is_missing_cache_op(exc, "concat_and_cache_mla"):
                        raise
                    native_cache_op_available = False
                    logger.warning(
                        "Vendor vLLM has no _C_cache_ops.concat_and_cache_mla; "
                        "trying FlagGems and then the BF16 correctness fallback"
                    )

            if not flaggems_cache_writer_loaded:
                flaggems_cache_writer_loaded = True
                try:
                    from flag_gems.fused.concat_and_cache_mla import (
                        concat_and_cache_mla as flaggems_cache_writer_impl,
                    )

                    flaggems_cache_writer = flaggems_cache_writer_impl
                except (ImportError, AttributeError, OSError):
                    flaggems_cache_writer = None

            if flaggems_cache_writer is not None:
                try:
                    flag_cache_dtype = (
                        "auto"
                        if kv_cache_dtype == "bfloat16"
                        else kv_cache_dtype
                    )
                    return flaggems_cache_writer(
                        kv_c,
                        k_pe,
                        kv_cache,
                        slot_mapping,
                        kv_cache_dtype=flag_cache_dtype,
                        scale=scale,
                    )
                except (NotImplementedError, RuntimeError) as exc:
                    logger.warning(
                        "FlagGems concat_and_cache_mla rejected this workload; "
                        "using the BF16 correctness fallback: %s",
                        exc,
                    )
                    flaggems_cache_writer = None

            return _concat_and_cache_mla_bf16_fallback(
                kv_c,
                k_pe,
                kv_cache,
                slot_mapping,
                kv_cache_dtype,
                scale,
            )

        concat_and_cache_mla_vendor_first._glm5_next_vendor_fallback = True
        concat_and_cache_mla_vendor_first._glm5_next_original = (
            concat_and_cache_mla
        )
        custom_ops.concat_and_cache_mla = concat_and_cache_mla_vendor_first
        installed = True

    return installed


def _silu_and_mul_with_clamp_oot(self, x: torch.Tensor) -> torch.Tensor:
    """Use FlagGems for GLM's bounded SwiGLU when its exact op is present."""
    if self.alpha != 1.0 or self.beta != 0.0:
        return self.forward_native(x)
    dim = x.shape[-1] // 2
    try:
        from flag_gems.fused.silu_and_mul_with_clamp import (
            silu_and_mul_with_clamp,
        )

        return silu_and_mul_with_clamp(
            x[..., :dim], x[..., dim:], self.swiglu_limit
        )
    except (ImportError, OSError, NotImplementedError, RuntimeError):
        return self.forward_native(x)


def _mhc_rms_norm(
    layer_input: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> torch.Tensor:
    """Apply the RMSNorm fused by the CUDA mHC reference kernels."""
    if norm_weight is None:
        return layer_input
    layer_input_fp32 = layer_input.float()
    inv_rms = torch.rsqrt(
        layer_input_fp32.square().mean(dim=-1, keepdim=True) + norm_eps
    )
    return (layer_input_fp32 * inv_rms * norm_weight.float()).to(
        layer_input.dtype
    )


def _mhc_pre_oot_with_norm(
    self,
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    norm_weight=None,
    norm_eps=0.0,
):
    try:
        from flag_gems.fused.mhc import mhc_pre

        post_mix, comb_mix, layer_input = mhc_pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
        )
    except (ImportError, OSError, NotImplementedError, RuntimeError):
        post_mix, comb_mix, layer_input = self.forward_native(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            norm_weight,
            norm_eps,
        )
    return (
        post_mix,
        comb_mix,
        _mhc_rms_norm(layer_input, norm_weight, norm_eps),
    )


def _mhc_post_oot_flaggems(self, x, residual, post_layer_mix, comb_res_mix):
    try:
        from flag_gems.fused.mhc import mhc_post

        return mhc_post(x, residual, post_layer_mix, comb_res_mix)
    except (ImportError, OSError, NotImplementedError, RuntimeError):
        return self.forward_native(x, residual, post_layer_mix, comb_res_mix)


def _mhc_fused_post_pre_oot_with_norm(
    self,
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    tile_n=1,
    norm_weight=None,
    norm_eps=0.0,
):
    try:
        from flag_gems.fused.mhc import mhc_post, mhc_pre

        residual_cur = mhc_post(x, residual, post_layer_mix, comb_res_mix)
        post_mix, comb_mix, layer_input = mhc_pre(
            residual_cur,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
        )
    except (ImportError, OSError, NotImplementedError, RuntimeError):
        residual_cur, post_mix, comb_mix, layer_input = self.forward_native(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_weight,
            norm_eps,
        )
    return (
        residual_cur,
        post_mix,
        comb_mix,
        _mhc_rms_norm(layer_input, norm_weight, norm_eps),
    )


def _mhc_pre_oot_bounded_cuda(self, residual, *args, **kwargs):
    if residual.shape[0] <= _MHC_CUDA_MAX_TOKENS:
        return self.forward_cuda(residual, *args, **kwargs)
    return _mhc_pre_oot_with_norm(self, residual, *args, **kwargs)


def _mhc_post_oot_bounded_cuda(self, x, residual, *args, **kwargs):
    if x.shape[0] <= _MHC_CUDA_MAX_TOKENS:
        return self.forward_cuda(x, residual, *args, **kwargs)
    return self.forward_native(x, residual, *args, **kwargs)


def _mhc_fused_post_pre_oot_bounded_cuda(
    self, x, residual, *args, **kwargs
):
    if x.shape[0] <= _MHC_CUDA_MAX_TOKENS:
        return self.forward_cuda(x, residual, *args, **kwargs)
    return _mhc_fused_post_pre_oot_with_norm(
        self, x, residual, *args, **kwargs
    )


class Glm5NextModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Preserve VLM checkpoints and default bare text configs to CausalLM."""

    def get_architectures(self) -> list[str]:
        architectures = super().get_architectures()
        if not architectures:
            architectures = [_CAUSAL_ARCH]
        self.hf_config.architectures = architectures.copy()
        return architectures


class Glm5NextForCausalLMConfig(HybridAttentionMambaModelConfig):
    """Compose v0.24 hybrid-cache and DSA validation."""

    @classmethod
    def verify_and_update_config(cls, vllm_config) -> None:
        HybridAttentionMambaModelConfig.verify_and_update_config(vllm_config)

        text_config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        if cache_config.cache_dtype == "bfloat16":
            cache_config.cache_dtype = "auto"
        if getattr(text_config, "index_kpool_compress", False):
            kpool = int(getattr(text_config, "index_kpool", 1))
            required = kpool * 32
            if cache_config.block_size % required:
                logger.info(
                    "GLM5-Next kpool changes KV block_size from %d to %d "
                    "for a 32-entry DeepGEMM compressed page",
                    cache_config.block_size,
                    required,
                )
                cache_config.block_size = required


def apply_glm5_next_v024_patches() -> bool:
    """Install idempotent config, convertor, and lazy-model registrations."""
    if not is_vllm_024():
        return False

    from vllm.platforms import current_platform
    from vllm.model_executor.models import config as model_config
    from vllm.model_executor.models import registry as model_registry
    from vllm.transformers_utils import config as transformers_config
    from vllm.transformers_utils import model_arch_config_convertor
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerBackend,
    )

    from vllm_fl.configs.glm5_next import (
        Glm5NextConfig,
        Glm5NextTextConfig,
        Glm5NextVisionConfig,
    )
    from vllm_fl.patches.glm5_next_kpool_v024 import (
        install_glm5_next_kpool_v024,
    )

    install_glm5_next_kpool_v024()

    # vLLM dispatches every CustomOp through ``forward_oot`` when an OOT
    # platform plugin is active.  Its default OOT implementation delegates to
    # ``forward_native``.  That fallback is not semantically interchangeable
    # for mHC: MHCPreOp and MHCFusedPostPreOp accept the fused RMSNorm weight
    # and epsilon, while their torch fallbacks intentionally ignore both.
    # GLM5-Next relies on that fused normalization twice per decoder layer, so
    # taking the fallback removes the layer norms and destroys model output.
    #
    # NVIDIA FlagOS ships the same CUDA/TileLang mHC operators used by native
    # vLLM, so bind its OOT entry points directly to the reference kernels.
    # Other OOT devices retain the portable torch decomposition, but complete
    # its missing fused RMSNorm explicitly.  Install either path before model
    # construction caches CustomOp._forward_method.
    if current_platform.is_out_of_tree():
        from vllm.model_executor.layers.mhc import (
            MHCFusedPostPreOp,
            MHCPostOp,
            MHCPreOp,
        )

        if use_nvidia_reference():
            global _MHC_CUDA_MAX_TOKENS
            _MHC_CUDA_MAX_TOKENS = int(
                os.environ.get("VLLM_FL_GLM5_MHC_CUDA_MAX_TOKENS", "0")
            )
            if _MHC_CUDA_MAX_TOKENS > 0:
                MHCPreOp.forward_oot = _mhc_pre_oot_bounded_cuda
                MHCPostOp.forward_oot = _mhc_post_oot_bounded_cuda
                MHCFusedPostPreOp.forward_oot = (
                    _mhc_fused_post_pre_oot_bounded_cuda
                )
                logger.info(
                    "Bound GLM5-Next mHC OOT dispatch to CUDA/TileLang for "
                    "<=%d tokens and portable reference fallback above it",
                    _MHC_CUDA_MAX_TOKENS,
                )
            else:
                for mhc_op in (MHCPreOp, MHCPostOp, MHCFusedPostPreOp):
                    mhc_op.forward_oot = mhc_op.forward_cuda
                logger.info(
                    "Bound GLM5-Next mHC OOT dispatch to NVIDIA CUDA/TileLang "
                    "kernels"
                )
        else:
            from vllm.model_executor.layers.activation import SiluAndMulWithClamp

            MHCPreOp.forward_oot = _mhc_pre_oot_with_norm
            MHCPostOp.forward_oot = _mhc_post_oot_flaggems
            MHCFusedPostPreOp.forward_oot = (
                _mhc_fused_post_pre_oot_with_norm
            )
            SiluAndMulWithClamp.forward_oot = _silu_and_mul_with_clamp_oot
            logger.info(
                "Installed GLM5-Next portable mHC OOT fallback with RMSNorm "
                "and FlagGems bounded-SwiGLU dispatch"
            )

    config_registry = transformers_config._CONFIG_REGISTRY
    config_registry["glm5_next"] = Glm5NextConfig
    config_registry["glm5_next_text"] = Glm5NextTextConfig
    config_registry["glm5_next_vision"] = Glm5NextVisionConfig

    convertors = model_arch_config_convertor.MODEL_ARCH_CONFIG_CONVERTORS
    convertors["glm5_next"] = Glm5NextModelArchConfigConvertor
    convertors["glm5_next_text"] = Glm5NextModelArchConfigConvertor

    # The indexer cache is [num_blocks, block_size, head_bytes], and its CUDA
    # insertion/top-k paths read kv_cache.stride(0).  It therefore supports the
    # padded physical page view already implemented by vLLM 0.24, even though
    # the backend conservatively reports False (to reject cross-layer layout).
    # Opt in only to block-stride indexing; this does not enable cross-layer
    # cache sharing and requires no vLLM source mutation.
    DeepseekV32IndexerBackend.indexes_kv_by_block_stride = classmethod(
        lambda cls: True
    )

    # Preserve each platform's selected sparse-MLA backend, including faster
    # vendor implementations, while filling two optional vLLM extension ABI
    # edges.  GLM5-Next is pure NoPE, so concat_mla_q is an exact direct copy.
    # Cache insertion remains vendor-first and only falls back when the vendor
    # wheel genuinely lacks _C_cache_ops.concat_and_cache_mla.
    _install_mla_boundary_compat_ops()

    for architecture in (_CAUSAL_ARCH, _CONDITIONAL_ARCH):
        model_config.MODELS_CONFIG_MAP[architecture] = Glm5NextForCausalLMConfig

    model_registry._TEXT_GENERATION_MODELS.setdefault(
        _CAUSAL_ARCH, ("glm5_next", _CAUSAL_ARCH)
    )
    model_registry._VLLM_MODELS.setdefault(
        _CAUSAL_ARCH, ("glm5_next", _CAUSAL_ARCH)
    )
    model_registry.ModelRegistry.register_model(
        _CAUSAL_ARCH,
        f"vllm_fl.models.glm5_next:{_CAUSAL_ARCH}",
    )

    # Keep the checkpoint's conditional architecture so vLLM constructs the
    # vision tower and enables --mm-encoder-tp-mode data instead of silently
    # reducing the model to its text-only runtime.
    model_registry._VLLM_MODELS.setdefault(
        _CONDITIONAL_ARCH, ("glm5_next", _CONDITIONAL_ARCH)
    )
    model_registry.ModelRegistry.register_model(
        _CONDITIONAL_ARCH,
        "vllm_fl.models.glm5_next_multimodal:"
        f"{_CONDITIONAL_ARCH}",
    )

    logger.info(
        "Installed vLLM 0.24 GLM5-Next text/VLM runtime with bounded KDA "
        "gate, kpool, ViT data parallelism, and provider=%s",
        get_glm5_provider(),
    )
    return True


__all__ = [
    "Glm5NextForCausalLMConfig",
    "Glm5NextModelArchConfigConvertor",
    "apply_glm5_next_v024_patches",
]
