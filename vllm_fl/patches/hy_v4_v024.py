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

    vLLM's fa_utils.py only defines flash_attn_varlen_func for CUDA/XPU/ROCm
    at module level. On MetaX the name doesn't exist in fa_utils, which causes
    downstream imports (mla/prefill/flash_attn.py) to fail with ImportError.

    Fix: inject flash_attn_varlen_func into fa_utils BEFORE patching the
    availability check, so that subsequent imports find the symbol.
    """
    from vllm.platforms import current_platform

    if current_platform.is_cuda() or current_platform.is_xpu():
        return

    # Check if MetaX flash_attn is actually available
    try:
        from flash_attn import flash_attn_varlen_func as _fa_varlen
    except ImportError:
        logger.warning(
            "flash_attn not available on this platform, "
            "MLA prefill backend will use torch SDPA fallback"
        )
        _patch_mla_prefill_torch_fallback()
        return

    try:
        from vllm.v1.attention.backends import fa_utils

        # Step 1: inject the function into fa_utils so that
        # `from fa_utils import flash_attn_varlen_func` works
        fa_utils.flash_attn_varlen_func = _fa_varlen

        # Step 2: patch the availability check
        fa_utils.is_flash_attn_varlen_func_available = lambda: True

        logger.info(
            "Patched fa_utils: injected flash_attn_varlen_func and "
            "is_flash_attn_varlen_func_available() for MetaX platform"
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


def _register_mx_sparse_attn_indexer_op() -> bool:
    """Load and register the MetaX sparse_attn_indexer op from vllm_metax.

    This loads vllm_metax's fp8.py module which registers
    torch.ops.vllm.mx_sparse_attn_indexer (and bf16 variant). These ops use
    MetaX's own compiled kernels for indexer cache read/write, MQA logits,
    and top-k selection.
    """
    import importlib.util
    import sys
    import torch

    if hasattr(torch.ops.vllm, "mx_sparse_attn_indexer"):
        return True  # already registered

    # The MetaX fp8.py imports DeepseekV32IndexerMetadata from
    # vllm_metax.v1.attention.backends.mla.indexer, but in FL plugin mode
    # the actual attn_metadata uses the class from the upstream vllm module.
    # We alias the metax module to the upstream one so isinstance checks pass.
    upstream_indexer_mod = "vllm.v1.attention.backends.mla.indexer"
    metax_indexer_mod = "vllm_metax.v1.attention.backends.mla.indexer"
    if upstream_indexer_mod in sys.modules and metax_indexer_mod not in sys.modules:
        sys.modules[metax_indexer_mod] = sys.modules[upstream_indexer_mod]
        # Ensure parent packages exist in sys.modules for the alias to work
        for parent in [
            "vllm_metax.v1",
            "vllm_metax.v1.attention",
            "vllm_metax.v1.attention.backends",
            "vllm_metax.v1.attention.backends.mla",
        ]:
            if parent not in sys.modules:
                import types
                sys.modules[parent] = types.ModuleType(parent)

    base_path = "/opt/conda/lib/python3.12/site-packages/vllm_metax/customized/layers/sparse_attn_indexer"
    for filename, mod_name in [
        ("fp8.py", "vllm_metax_indexer_fp8"),
        ("bf16.py", "vllm_metax_indexer_bf16"),
    ]:
        path = f"{base_path}/{filename}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            return False
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return hasattr(torch.ops.vllm, "mx_sparse_attn_indexer")


def _patch_bf16_paged_mqa_logits_debug() -> None:
    """Wrap vllm_metax's bf16_paged_mqa_logits to dump decode-time inputs.

    The decode-phase indexer kernel bf16_paged_mqa_logits_scheduled_kernel
    hit a MACA hardware trap (Xnack/ATU Fault = out-of-bounds device read).
    An ATU fault in a paged kernel almost always means block_tables points
    past the physically allocated KV blocks, context_lens/schedule_metadata
    disagree with the cache, or the cache stride the kernel assumes differs
    from how we allocated it. Its own docstring claims the cache should be
    [num_blocks, block_size, 1, D+4] uint8 (packed bf16+scale), which is at
    odds with our plain-bf16 [.., D] allocation — this log lets us confirm
    the real shapes/strides and whether block_tables exceeds num_blocks
    before the kernel dereferences them.

    One-shot (first decode call) to avoid per-step spam.
    """
    import torch

    try:
        from vllm_metax.utils import deep_gemm as _mx_dg
    except Exception as e:  # pragma: no cover - metax not present
        logger.warning(f"Cannot import vllm_metax.utils.deep_gemm for debug: {e}")
        return

    _orig = getattr(_mx_dg, "bf16_paged_mqa_logits", None)
    if _orig is None or getattr(_orig, "_hy4_debug_wrapped", False):
        return

    @wraps(_orig)
    def _wrapped(q_bf16, kv_cache_bf16, weights, context_lens, block_tables,
                 schedule_metadata, max_model_len, *args, **kwargs):
        if not getattr(_wrapped, "_logged", False):
            _wrapped._logged = True
            try:
                bt = block_tables
                cl = context_lens
                num_blocks = kv_cache_bf16.shape[0] if kv_cache_bf16.ndim >= 1 else None
                bt_max = int(bt.max().item()) if bt is not None and bt.numel() else None
                bt_min = int(bt.min().item()) if bt is not None and bt.numel() else None
                cl_max = int(cl.max().item()) if cl is not None and cl.numel() else None
                logger.info(
                    "[hy4-paged-mqa] q_bf16=%s/%s kv_cache=%s/%s "
                    "(num_blocks=%s block_size=%s last2=%s) weights=%s/%s "
                    "context_lens=%s/%s(max=%s) block_tables=%s/%s(min=%s max=%s) "
                    "schedule_metadata=%s/%s max_model_len=%s | "
                    "OOB CHECK: block_tables.max(%s) must be < num_blocks(%s)",
                    tuple(q_bf16.shape), q_bf16.dtype,
                    tuple(kv_cache_bf16.shape), kv_cache_bf16.dtype,
                    num_blocks,
                    kv_cache_bf16.shape[1] if kv_cache_bf16.ndim >= 2 else None,
                    tuple(kv_cache_bf16.shape[-2:]) if kv_cache_bf16.ndim >= 2 else None,
                    tuple(weights.shape), weights.dtype,
                    tuple(cl.shape), cl.dtype, cl_max,
                    tuple(bt.shape), bt.dtype, bt_min, bt_max,
                    tuple(schedule_metadata.shape) if schedule_metadata is not None else None,
                    schedule_metadata.dtype if schedule_metadata is not None else None,
                    max_model_len,
                    bt_max, num_blocks,
                )
                if bt_max is not None and num_blocks is not None and bt_max >= num_blocks:
                    logger.warning(
                        "[hy4-paged-mqa] block_tables.max(%s) >= num_blocks(%s) "
                        "-> paged kernel WILL read OOB (this is the ATU Fault).",
                        bt_max, num_blocks,
                    )
            except Exception as le:
                logger.warning(f"[hy4-paged-mqa] debug logging failed: {le}")
        return _orig(q_bf16, kv_cache_bf16, weights, context_lens, block_tables,
                     schedule_metadata, max_model_len, *args, **kwargs)

    _wrapped._hy4_debug_wrapped = True
    _mx_dg.bf16_paged_mqa_logits = _wrapped
    # bf16.py imported the symbol by value, and it is loaded under the custom
    # module name "vllm_metax_indexer_bf16" (see _register_mx_sparse_attn_indexer_op,
    # which execs it via spec_from_file_location). Do NOT `import` the real
    # package path — that triggers bf16.py's transitive imports which fail in
    # FL plugin mode (vllm_metax.v1.attention.ops missing). Patch the already
    # loaded module object directly from sys.modules; if it isn't loaded yet,
    # the source-module patch above still covers late imports.
    import sys
    patched_here = False
    for _name, _m in list(sys.modules.items()):
        if _m is None:
            continue
        if getattr(_m, "bf16_paged_mqa_logits", None) is _orig:
            _m.bf16_paged_mqa_logits = _wrapped
            patched_here = True
            logger.info("[hy4-paged-mqa] patched symbol in module %s", _name)
    if not patched_here:
        logger.info(
            "[hy4-paged-mqa] bf16 module not loaded yet; source-module patch "
            "on deep_gemm will cover it"
        )
    logger.info("[hy4-paged-mqa] wrapped bf16_paged_mqa_logits for one-shot debug")


def _patch_sparse_attn_indexer_for_maca() -> None:
    """Patch SparseAttnIndexer.forward_native to use MetaX kernels.

    The upstream forward_cuda calls NVIDIA-only C++ ops
    (indexer_k_quant_and_cache with fp8_e4m3, DeepGEMM fp8_mqa_logits, etc.)
    that are not available on MACA. MetaX provides equivalent ops via
    torch.ops.vllm.mx_sparse_attn_indexer which handles bf16 keys natively
    and uses MACA-compiled kernels for cache, logits, and top-k.
    """
    import torch
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import _encode_layer_name

    if current_platform.is_cuda():
        return  # real NVIDIA, no patch needed

    try:
        from vllm.model_executor.layers.sparse_attn_indexer import (
            SparseAttnIndexer,
        )

        if not _register_mx_sparse_attn_indexer_op():
            logger.warning(
                "Failed to register mx_sparse_attn_indexer op; "
                "SparseAttnIndexer will not work on MACA"
            )
            return

        _orig_forward_native = SparseAttnIndexer.forward_native

        def _forward_native_maca(self, hidden_states, q_quant, k, weights):
            if not current_platform.is_cuda_alike():
                return _orig_forward_native(
                    self, hidden_states, q_quant, k, weights
                )

            # MACA only has bf16_mqa_logits — always use bf16 op
            if isinstance(q_quant, tuple):
                q_values, q_scale = q_quant
            else:
                q_values, q_scale = q_quant, None

            if q_values.dtype not in (torch.bfloat16, torch.float16):
                q_values = q_values.to(torch.bfloat16)
                q_scale = None

            # One-shot debug: dump the exact shapes/dtypes handed to the bf16
            # MQA-logits kernel. deep_gemm/bf16_attention.py:366 asserts
            # ``kv_heads == 1 and kv_dim == head_dim``; the kv_cache last dim
            # (kv_dim) must equal self.head_dim (no packed fp8 scale). This log
            # tells us at a glance whether the cache layout fix took effect and,
            # if the assert still fires, which half fails. Gated on a per-op
            # class attr so it prints once (prefill + decode) not every step.
            if not getattr(SparseAttnIndexer, "_mx_bf16_shape_logged", False):
                kv = self.k_cache.kv_cache
                logger.info(
                    "[hy4-indexer-bf16] q_values=%s/%s k=%s/%s kv_cache=%s/%s "
                    "head_dim=%s quant_block_size=%s scale_fmt=%s topk=%s "
                    "q_scale=%s skip_k_cache_insert=%s use_fp4_cache=%s | "
                    "expect kv_cache last dim == head_dim (%s)",
                    tuple(q_values.shape), q_values.dtype,
                    tuple(k.shape) if k is not None else None,
                    k.dtype if k is not None else None,
                    tuple(kv.shape), kv.dtype,
                    self.head_dim, self.quant_block_size, self.scale_fmt,
                    self.topk_tokens,
                    None if q_scale is None else (tuple(q_scale.shape), q_scale.dtype),
                    self.skip_k_cache_insert, self.use_fp4_cache,
                    self.head_dim,
                )
                if kv.ndim >= 1 and kv.shape[-1] != self.head_dim:
                    logger.warning(
                        "[hy4-indexer-bf16] kv_cache last dim %s != head_dim %s "
                        "-> bf16_paged_mqa_logits assert kv_dim==head_dim WILL "
                        "fail. Cache built with fp8-packed width? Check "
                        "hy_v4_attention.py Indexer k_cache layout.",
                        kv.shape[-1], self.head_dim,
                    )
                SparseAttnIndexer._mx_bf16_shape_logged = True

            return torch.ops.vllm.mx_sparse_attn_indexer_bf16(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_values,
                q_scale,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                self.skip_k_cache_insert,
                self.use_fp4_cache,
            )

        SparseAttnIndexer.forward_native = _forward_native_maca
        logger.info(
            "Patched SparseAttnIndexer.forward_native to use MetaX "
            "mx_sparse_attn_indexer op"
        )
        # Instrument the decode paged-logits kernel to catch the ATU Fault.
        _patch_bf16_paged_mqa_logits_debug()
    except Exception as e:
        logger.warning(f"Failed to patch SparseAttnIndexer: {e}")


def _patch_flashmla_sparse_for_metax() -> None:
    """Patch FlashMLA sparse decode to use MetaX kernels instead of vllm._flashmla_C.

    The native FLASHMLA_SPARSE backend requires NVIDIA's compiled _flashmla_C
    extension. MetaX has its own flash_mla library with sparse support. This
    patch redirects the kernel calls and fixes capability checks so both the
    registry-based path and HYV4FlashMLASparseBackend work on MACA.
    """
    from vllm.platforms import current_platform

    if current_platform.is_cuda():
        return  # real NVIDIA, no patch needed

    try:
        from vllm_fl.dispatch.backends.vendor.metax.impl.attention.ops.flashmla import (
            flash_mla_sparse_prefill,
        )

        def _maca_flash_mla_sparse_fwd(
            q, kv, indices, sm_scale, topk_length=None, attn_sink=None
        ):
            """Wrapper matching the native flash_mla_sparse_fwd signature."""
            return flash_mla_sparse_prefill(q, kv, indices, sm_scale)

        # Patch the ops module so any code importing from there gets MetaX version
        import vllm.v1.attention.ops.flashmla as flashmla_ops
        flashmla_ops.flash_mla_sparse_fwd = _maca_flash_mla_sparse_fwd

        # Patch the already-imported reference in the sparse backend module
        import vllm.v1.attention.backends.mla.flashmla_sparse as sparse_mod
        sparse_mod.flash_mla_sparse_fwd = _maca_flash_mla_sparse_fwd

        # Patch supports_compute_capability so HYV4FlashMLASparseBackend passes
        # validation (it inherits from the native FlashMLASparseBackend)
        from vllm.v1.attention.backends.mla.flashmla_sparse import (
            FlashMLASparseBackend,
        )

        @classmethod  # type: ignore[misc]
        def _maca_supports_capability(cls, capability) -> bool:
            return True

        FlashMLASparseBackend.supports_compute_capability = (
            _maca_supports_capability
        )

        # Also patch the hy_v4_flashmla_sparse module's imported symbols
        try:
            from vllm_fl.models import hy_v4_flashmla_sparse as hy4_sparse
            hy4_sparse.flash_mla_sparse_fwd = _maca_flash_mla_sparse_fwd
        except (ImportError, AttributeError):
            pass

        logger.info(
            "Patched FlashMLA sparse decode to use MetaX flash_mla kernels"
        )
    except Exception as e:
        logger.warning(f"Failed to patch FlashMLA sparse for MetaX: {e}")


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

    # Patch SparseAttnIndexer to work on MACA (is_cuda_alike but not is_cuda)
    _patch_sparse_attn_indexer_for_maca()

    # Patch FlashMLA sparse decode to use MetaX kernels
    _patch_flashmla_sparse_for_metax()

    logger.info("Installed HY4 runtime compatibility for vLLM 0.24")
    return True


__all__ = [
    "HYV4ModelArchConfigConvertor",
    "apply_hy_v4_v024_patches",
]
