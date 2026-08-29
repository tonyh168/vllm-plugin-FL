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

from vllm.forward_context import get_forward_context

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
        # Register in sys.modules BEFORE exec (the standard import contract):
        # module_from_spec + exec_module does not do this automatically, so the
        # module was previously invisible to sys.modules.get(mod_name). The
        # bf16 paged-mqa debug hook relies on finding "vllm_metax_indexer_bf16"
        # there to patch the by-value-imported bf16_paged_mqa_logits symbol
        # (bf16.py:188 calls it by bare name). Registering pre-exec also matches
        # CPython's own import machinery and avoids half-initialized modules on
        # any circular import.
        sys.modules[mod_name] = mod
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
                # WARNING level: this deployment filters INFO for the
                # vllm_fl.patches.hy_v4_v024 logger, so the whole point of the
                # hook (this dump) is invisible at INFO. Use WARNING so it lands.
                logger.warning(
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

                # --- CONTENT dump: root-cause hypothesis is that schedule_metadata
                # is UNINITIALISED garbage (torch.empty never filled on non-CUDA).
                # Shapes alone can't tell; we must inspect the actual VALUES.
                def _peek(t, n=16):
                    # Flatten to CPU and show first n values + basic stats without
                    # a full .cpu() sync of a huge tensor.
                    if t is None:
                        return "None"
                    try:
                        flat = t.detach().reshape(-1)
                        head = flat[:n].tolist()
                        tmin = int(flat.min().item())
                        tmax = int(flat.max().item())
                        return f"head{n}={head} min={tmin} max={tmax} numel={flat.numel()}"
                    except Exception as pe:
                        return f"<peek failed: {pe}>"

                logger.warning(
                    "[hy4-paged-mqa] VALUES: context_lens=%s | block_tables=%s | "
                    "schedule_metadata=%s",
                    _peek(cl), _peek(bt), _peek(schedule_metadata),
                )

                # NOTE: the "REF schedule_metadata recompute + MATCHES_PASSED_IN"
                # block was removed here. It derived num_sms from
                # schedule_metadata.shape[0]-1, but after the round-16 fix the
                # buffer is (num_blocks+1, 2)=(3329,2), so that reverse-derivation
                # produced num_sms=3328 and the recompute always reported a false
                # MISMATCH. Its job (confirming the round-12/13 dirty-metadata root
                # cause) is done; keeping it now only misleads. The VALUES dump
                # above still shows the actual passed-in schedule_metadata content.
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
    # Patch only the known modules that hold the symbol by value. Do NOT scan
    # all of sys.modules with getattr — some vLLM lazy modules define a custom
    # __getattr__(name) that raises when probed for an unknown attr, which would
    # crash this whole patch (and take the indexer forward_native patch with it).
    import sys
    patched_here = False
    # bf16.py is exec'd under the custom name "vllm_metax_indexer_bf16" by
    # _register_mx_sparse_attn_indexer_op (spec_from_file_location).
    for _name in ("vllm_metax_indexer_bf16",
                  "vllm_metax.customized.layers.sparse_attn_indexer.bf16"):
        _m = sys.modules.get(_name)
        if _m is None:
            continue
        try:
            if getattr(_m, "bf16_paged_mqa_logits", None) is _orig:
                _m.bf16_paged_mqa_logits = _wrapped
                patched_here = True
                logger.warning("[hy4-paged-mqa] patched symbol in module %s", _name)
        except Exception as pe:
            logger.warning("[hy4-paged-mqa] probe of %s failed: %s", _name, pe)
    if not patched_here:
        logger.warning(
            "[hy4-paged-mqa] bf16 module symbol NOT patched here (sys.modules "
            "miss); source-module patch on deep_gemm may NOT cover bf16.py's "
            "by-value import — decode dump may not fire"
        )
    # WARNING level on purpose: this deployment filters INFO for this logger.
    logger.warning("[hy4-paged-mqa] wrapped bf16_paged_mqa_logits for one-shot debug")


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
                # WARNING: INFO is filtered for this logger in the deployment.
                logger.warning(
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

            topk_indices = torch.ops.vllm.mx_sparse_attn_indexer_bf16(
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

            # Diagnose the "model repeats the last input token" garbage output
            # (log round 17). Symptom == attention aggregates ~nothing, which for
            # a DSA sparse model usually means the indexer's top-k selection is
            # broken (all -1 / all same pos / uninitialised), so attention only
            # ever sees one token.
            #
            # Round 20 subagent finding: the PREVIOUS gate ("first 4 calls") got
            # consumed entirely by the profiling / _dummy_run passes, where
            # get_forward_context().attn_metadata is NOT a dict. In that case the
            # metax op takes the fake/profiling branch (bf16.py:58) and returns
            # the uninitialised torch.empty buffer verbatim (bf16.py:244) — no -1
            # fill, no top-k. That is exactly the "huge +/- ints, neg1_count=0"
            # garbage we saw; it was a MEASUREMENT ARTIFACT of dummy runs, not a
            # real-inference bug. A REAL prefill always writes -1 first
            # (bf16.py:107) + pads short rows with -1, so real dumps must show a
            # LARGE neg1_count.
            #
            # New gate: only count/dump calls where attn_metadata is a real dict
            # (i.e. actual inference steps), and log the branch discriminator so
            # we can separate:
            #   H1 -> real steps DO run the real op (md_is_dict=True, sane spread
            #         + partial -1) => indexer is fine, bug is downstream
            #         (FlashMLA sparse decode / MLA aggregation).
            #   H2 -> even real steps see md_is_dict=False or prefix not in md
            #         keys => op keeps hitting the fake branch => real bug here.
            # Interpretation of the dumped indices when md_is_dict=True:
            #   sane spread + partial -1 -> top-k fine, look downstream
            #   all -1                   -> indexer selected no KV -> repeats
            #   all 0 / one pos          -> selection collapsed to a single pos
            #   huge +/- ints (neg1=0)   -> fake branch STILL taken (=> H2)
            try:
                _fc = get_forward_context()
                _md = getattr(_fc, "attn_metadata", None)
                _md_is_dict = isinstance(_md, dict)
                _enc_prefix = _encode_layer_name(self.k_cache.prefix)
                _prefix_in_md = bool(_md_is_dict and _enc_prefix in _md)
            except Exception:
                _md = None
                _md_is_dict = False
                _enc_prefix = None
                _prefix_in_md = False

            _n = getattr(SparseAttnIndexer, "_mx_topk_dump_count", 0)
            # Only spend the dump budget on REAL steps (attn_metadata is a dict).
            if _md_is_dict and _n < 6:
                SparseAttnIndexer._mx_topk_dump_count = _n + 1
                try:
                    ti = topk_indices
                    rows = min(3, ti.shape[0])
                    cols = min(12, ti.shape[-1])
                    sample = ti[:rows, :cols].detach().cpu().tolist()
                    flat = ti.detach().reshape(-1)
                    n_neg1 = int((flat == -1).sum().item())
                    logger.warning(
                        "[hy4-topk] REAL-STEP md_is_dict=%s prefix_in_md=%s "
                        "enc_prefix=%s md_type=%s md_keys=%s | shape=%s dtype=%s | "
                        "per-row[:%d,:%d]=%s | min=%s max=%s numel=%s "
                        "neg1_count=%s (%.1f%%) | phase=%s(q_rows=%s topk_tokens=%s)",
                        _md_is_dict, _prefix_in_md, _enc_prefix,
                        type(_md).__name__,
                        (list(_md.keys())[:4] if _md_is_dict else None),
                        tuple(ti.shape), ti.dtype, rows, cols, sample,
                        int(flat.min().item()), int(flat.max().item()),
                        flat.numel(), n_neg1,
                        100.0 * n_neg1 / max(1, flat.numel()),
                        "prefill" if q_values.shape[0] > 1 else "decode",
                        q_values.shape[0], self.topk_tokens,
                    )
                except Exception as te:
                    logger.warning("[hy4-topk] dump failed: %s", te)
            elif not _md_is_dict:
                # Fake/profiling branch — log once so we can confirm dummy runs
                # are being correctly skipped (and never counted against budget).
                if not getattr(SparseAttnIndexer, "_mx_topk_fake_logged", False):
                    SparseAttnIndexer._mx_topk_fake_logged = True
                    logger.warning(
                        "[hy4-topk] SKIP fake/profiling call: attn_metadata "
                        "type=%s (not dict) -> metax op returns uninitialised "
                        "buffer; not counted. enc_prefix=%s",
                        type(_md).__name__, _enc_prefix,
                    )

            return topk_indices

        SparseAttnIndexer.forward_native = _forward_native_maca
        logger.info(
            "Patched SparseAttnIndexer.forward_native to use MetaX "
            "mx_sparse_attn_indexer op"
        )
    except Exception as e:
        logger.warning(f"Failed to patch SparseAttnIndexer: {e}")

    # Instrument the decode paged-logits kernel to catch the ATU Fault.
    # Isolated in its own try so a debug-hook failure can never break the
    # (already applied) forward_native patch above.
    try:
        _patch_bf16_paged_mqa_logits_debug()
    except Exception as e:
        logger.warning(f"[hy4-paged-mqa] debug hook install failed (ignored): {e}")


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

        import os as _os
        # Round 24: wire topk_length -> indices_all_valid_per_q when set. The
        # native wrapper dropped both topk_length and attn_sink; short prompts
        # have ~2044/2048 padded (-1) lanes, so if the MetaX kernel needs the
        # explicit valid-count to mask them, dropping it collapses softmax over
        # garbage -> "repeat last token". Default off = original behaviour.
        _pass_valid_len = _os.environ.get(
            "VLLM_HY4_SPARSE_PASS_VALIDLEN", "0"
        ) == "1"

        def _maca_flash_mla_sparse_fwd(
            q, kv, indices, sm_scale, topk_length=None, attn_sink=None
        ):
            """Wrapper matching the native flash_mla_sparse_fwd signature.

            Round 24 diagnostic: dump — once per real step — whether the native
            contract args (topk_length / attn_sink) are being dropped, plus the
            per-query valid-index count and the kernel output norm. This is the
            decisive probe for the "repeat last token" garbage.
            """
            valid_len = None
            if _pass_valid_len and topk_length is not None:
                valid_len = topk_length

            out = flash_mla_sparse_prefill(
                q, kv, indices, sm_scale,
                indices_all_valid_per_q=valid_len,
            )

            n = getattr(_maca_flash_mla_sparse_fwd, "_dbg_n", 0)
            if n < 6:
                _maca_flash_mla_sparse_fwd._dbg_n = n + 1
                try:
                    o = out[0] if isinstance(out, (tuple, list)) else out
                    # per-row valid index count (indices: [s_q, 1, topk])
                    idx2d = indices.reshape(indices.shape[0], -1)
                    valid_per_row = (idx2d >= 0).sum(dim=1)
                    last_out = o.reshape(o.shape[0], -1)[-1].float()
                    logger.warning(
                        "[hy4-mla-kernel] q=%s/%s indices=%s sm_scale=%s | "
                        "DROPPED topk_length=%s attn_sink=%s pass_validlen=%s | "
                        "valid_per_row[first3]=%s [last]=%s (topk=%s) | "
                        "out=%s/%s out_norm_per_tok[first]=%.4f [last]=%.4f "
                        "last_out_isnan=%s isinf=%s",
                        tuple(q.shape), q.dtype, tuple(indices.shape), sm_scale,
                        None if topk_length is None else tuple(topk_length.shape),
                        None if attn_sink is None else tuple(attn_sink.shape),
                        _pass_valid_len,
                        valid_per_row[:3].tolist(),
                        int(valid_per_row[-1].item()), idx2d.shape[1],
                        tuple(o.shape), o.dtype,
                        float(o.reshape(o.shape[0], -1)[0].float().norm().item()),
                        float(last_out.norm().item()),
                        bool(last_out.isnan().any().item()),
                        bool(last_out.isinf().any().item()),
                    )
                except Exception as _e:
                    logger.warning("[hy4-mla-kernel] dump failed: %s", _e)
            return out

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

        # Round 24c (moved from a local vllm edit so it propagates to every ray
        # rank via git, not just the head node): wrap _forward_bf16_kv to dump
        # the per-request topk valid-count BEFORE vs AFTER
        # triton_convert_req_index_to_global_index. The indexer emits correct
        # causal top-k (round 21), but [hy4-mla-kernel] saw the CONVERTED indices
        # collapse to mostly -1 (valid_per_row=[1,0,1,...,0]). This probe tells us
        # whether the global-slot conversion is what destroys the indices on MetaX.
        try:
            from vllm.v1.attention.backends.mla.flashmla_sparse import (
                FlashMLASparseImpl,
            )

            if not getattr(FlashMLASparseImpl, "_hy4_convert_wrapped", False):
                _orig_bf16 = FlashMLASparseImpl._forward_bf16_kv

                def _wrapped_forward_bf16_kv(
                    self, q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
                ):
                    n = getattr(FlashMLASparseImpl, "_hy4_cv_dbg_n", 0)
                    if n < 4:
                        try:
                            import torch as _t
                            # Round 24e: decisive H1 (write/read race) test. Read
                            # the SAME buffer slice twice: (1) NO sync — how the
                            # downstream triton kernel actually sees it; (2) after
                            # an explicit device sync — the settled value. If (1)
                            # is [1,0,1,0] but (2) becomes [1,2,3,4], the indexer
                            # write is not ordered before this read on MetaX =>
                            # race confirmed, fix = barrier the indexer op.
                            in2d = topk_indices.reshape(topk_indices.shape[0], -1)
                            valid_nosync = (in2d >= 0).sum(dim=1)[:6].tolist()
                            if _t.cuda.is_available():
                                _t.cuda.synchronize()
                            valid_synced = (
                                (topk_indices.reshape(topk_indices.shape[0], -1) >= 0)
                                .sum(dim=1)[:6].tolist()
                            )
                            # also snapshot first row's first 12 cols post-sync
                            row_sample = (
                                topk_indices.reshape(topk_indices.shape[0], -1)[:2, :12]
                                .detach().cpu().tolist()
                            )
                            logger.warning(
                                "[hy4-convert] BEFORE topk_indices=%s "
                                "valid_NOSYNC[:6]=%s valid_SYNCED[:6]=%s "
                                "row_sample[:2,:12]=%s min=%s max=%s | "
                                "req_id_per_token=%s block_table=%s block_size=%s",
                                tuple(topk_indices.shape),
                                valid_nosync, valid_synced, row_sample,
                                int(in2d.min().item()), int(in2d.max().item()),
                                tuple(attn_metadata.req_id_per_token.shape),
                                tuple(attn_metadata.block_table.shape),
                                attn_metadata.block_size,
                            )
                        except Exception as _e:
                            logger.warning("[hy4-convert] BEFORE dump failed: %s", _e)

                    out = _orig_bf16(
                        self, q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
                    )

                    # Re-run only the conversion to observe its output (cheap;
                    # the real one already ran inside _orig_bf16). Guarded + capped.
                    if n < 4:
                        FlashMLASparseImpl._hy4_cv_dbg_n = n + 1
                        try:
                            from vllm.v1.attention.backends.mla.sparse_utils import (
                                triton_convert_req_index_to_global_index,
                            )
                            conv_idx, conv_len = (
                                triton_convert_req_index_to_global_index(
                                    attn_metadata.req_id_per_token,
                                    attn_metadata.block_table,
                                    topk_indices,
                                    BLOCK_SIZE=attn_metadata.block_size,
                                    NUM_TOPK_TOKENS=topk_indices.shape[1],
                                    return_valid_counts=True,
                                )
                            )
                            o2d = conv_idx.reshape(conv_idx.shape[0], -1)
                            o_valid = (o2d >= 0).sum(dim=1)
                            logger.warning(
                                "[hy4-convert] AFTER topk_indices=%s "
                                "valid_per_row[:6]=%s topk_length[:6]=%s "
                                "min=%s max=%s",
                                tuple(conv_idx.shape), o_valid[:6].tolist(),
                                conv_len[:6].tolist() if conv_len is not None else None,
                                int(o2d.min().item()), int(o2d.max().item()),
                            )
                        except Exception as _e:
                            logger.warning("[hy4-convert] AFTER dump failed: %s", _e)
                    return out

                FlashMLASparseImpl._forward_bf16_kv = _wrapped_forward_bf16_kv
                FlashMLASparseImpl._hy4_convert_wrapped = True
                logger.warning("[hy4-convert] wrapped _forward_bf16_kv for probe")
        except Exception as _e:
            logger.warning("[hy4-convert] wrap install failed (ignored): %s", _e)

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

    # Fill decode schedule_metadata on MetaX (upstream is_cuda() gate leaves the
    # buffer uninitialised -> paged MQA kernel reads garbage -> ATU Fault).
    # Applied here (register_model time) rather than in glm_moe_dsa's
    # apply_platform_patches (register time) to avoid a circular import of
    # vllm.utils.torch_utils that fails at the earlier init stage.
    try:
        from vllm_fl.patches.glm_moe_dsa import patch_indexer_schedule_metadata
        patch_indexer_schedule_metadata()
    except Exception as e:
        logger.warning("[hy4-sched-meta] failed to apply schedule_metadata "
                       "patch: %s", e)

    logger.info("Installed HY4 runtime compatibility for vLLM 0.24")
    return True


__all__ = [
    "HYV4ModelArchConfigConvertor",
    "apply_hy_v4_v024_patches",
]
