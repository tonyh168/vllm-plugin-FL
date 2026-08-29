# Copyright (c) 2026 BAAI. All rights reserved.

"""
METAX backend implementation.

This backend provides operator implementations for METAX GPUs.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from vllm.logger import init_logger

from vllm_fl.dispatch.backends.base import Backend

from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    _ATTN_OVERRIDES,
    register_backend,
)

logger = init_logger(__name__)

# Per-process guard: register once and log once per process (incl. each spawned
# Ray worker). Repeat calls (e.g. lazy call from attention_backend()) are no-ops
# but still cheap; the flag only suppresses duplicate logging.
_ATTN_BACKENDS_REGISTERED = False


# Register attention backends for MACA
def register_attention_backends():
    global _ATTN_BACKENDS_REGISTERED

    register_backend(
        AttentionBackendEnum.FLASHMLA,
        class_path="vllm_fl.dispatch.backends.vendor.metax.impl.attention.mla.flashmla.MacaFlashMLABackend",
    )
    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        class_path="vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn.MacaFlashAttentionBackend",
    )
    # FLASHMLA_SPARSE: GLM5.3-Flash uses sparse MLA (deepseek_sparse_attention,
    # index_topk). Without a registration, the sparse path falls through to vLLM
    # upstream's flashmla_sparse.py FlashMLASparseImpl, which hard-requires the
    # compiled extension `vllm._flashmla_C` (absent on MetaX) and crashes with
    # "vllm._flashmla_C is not available".
    #
    # We target the vendor vllm_metax sparse MLA backend rather than plugin-FL's
    # own FlagGemsSparseMLABackend. Rationale (correcting an earlier assumption):
    #   - The FlagGems backend dispatches the sparse kernel through
    #     flag_gems.fused.flashmla_sparse.flash_mla_sparse_fwd. That kernel was
    #     never end-to-end validated on MetaX for accuracy; to make it merely
    #     RUN we had to strip its autotune configs down to a single minimal
    #     BK=16/BH=16 tile ("only-make-it-run, performance be damned"). Simple
    #     prompts return correct tokens, but GPQA (long context + fine-grained
    #     topk) surfaced accuracy problems traced to this path.
    #   - vllm_metax's MacaFlashMLASparseBackend dispatches through
    #     vllm_metax.v1.attention.ops.flashmla -> the `flash_mla` python package
    #     (flash_mla_cuda C extension), a MetaX-vendor-compiled & validated
    #     kernel. It does NOT touch FlagGems and does NOT need `vllm._flashmla_C`.
    #   - The "vllm_metax is a mutually-exclusive platform plugin" concern does
    #     NOT apply here: register_backend only writes a class-path string into
    #     _ATTN_OVERRIDES and lazily imports ONE backend class. It does not
    #     activate the vllm_metax platform, so it is compatible with
    #     VLLM_PLUGINS=fl. Verified offline: under VLLM_PLUGINS=fl the class
    #     resolves, get_supported_head_sizes()=[512,576] (GLM5.3 is 576),
    #     is_mla/is_sparse=True, and its impl reads indexer.topk_indices_buffer
    #     exactly like the FlagGems impl (interface-compatible).
    #
    # We register a thin plugin-FL subclass (Glm53FlashMLASparseBackend) rather
    # than the vendor class directly. It reuses the vendor's kernels/impl verbatim
    # but swaps in a metadata builder that forces bf16_use_mixed_batch=True. This
    # is required because GLM5.3-Flash is a use-nope MLA (qk_rope_head_dim=0 ->
    # head_size=512, not DeepSeek V3.2's 576): the vendor's default bf16 decode
    # path calls flash_mla_cuda.fwd_kvcache_mla, whose kernel hard-asserts
    # head_size==576 and crashes on 512. mixed_batch routes decode through the
    # sparse prefill kernel (flash_mla_sparse_fwd), which handles 512. Keeping the
    # override target inside plugin-FL avoids patching the non-version-controlled
    # vllm_metax conda package on every machine/reinstall.
    register_backend(
        AttentionBackendEnum.FLASHMLA_SPARSE,
        class_path="vllm_fl.dispatch.backends.vendor.metax.impl.attention.mla.flashmla_sparse.Glm53FlashMLASparseBackend",
    )

    # Defeat @cache poisoning on _cached_get_attn_backend: if the sparse MLA
    # layer resolved its backend BEFORE this override was written (prior rounds
    # crashed in upstream flashmla_sparse.py even though _ATTN_OVERRIDES showed
    # our override — proof the class was resolved/cached first), the @cache holds
    # the stale upstream class. Clearing it forces re-resolution through
    # get_path() -> our override on the next get_attn_backend() call. Safe: the
    # cache only memoizes the (backend, config, num_heads) -> class mapping.
    try:
        from vllm.v1.attention.selector import _cached_get_attn_backend

        _cached_get_attn_backend.cache_clear()
    except Exception as e:  # pragma: no cover - defensive
        print(
            f"[metax-attn-reg] cache_clear on _cached_get_attn_backend failed: {e!r}",
            flush=True,
        )

    # Log once per process what actually landed in _ATTN_OVERRIDES. This is the
    # ground truth the backend selector reads via AttentionBackendEnum.get_path()
    # -> get_class(). Interpretation for the next iteration:
    #   - line missing entirely  -> registration never ran in this process.
    #   - line shows FLASHMLA_SPARSE=...FlagGemsSparseMLABackend AND crash is gone
    #     -> override + cache_clear worked.
    #   - line shows our override but crash STILL in upstream flashmla_sparse.py
    #     FlashMLASparseImpl -> resolution happened after cache_clear again
    #     (some later code re-cached upstream) OR MLAAttention received an
    #     explicit attn_backend= and bypassed get_attn_backend() entirely
    #     (mla_attention.py:397 `if attn_backend is not None`). Next step then is
    #     to trace the construction site, not the registry.
    if not _ATTN_BACKENDS_REGISTERED:
        _ATTN_BACKENDS_REGISTERED = True
        # Use print(flush=True), not logger: prior rounds proved the platform
        # logger can be swallowed by level/Ray capture (the version banner had
        # to switch to print for the same reason). This must be visible so the
        # next iteration can tell "registration ran + what it wrote" apart from
        # "registration never ran in this process".
        print(
            "[metax-attn-reg] registered MLA backend overrides: "
            f"FLASHMLA={_ATTN_OVERRIDES.get(AttentionBackendEnum.FLASHMLA)} "
            f"FLASH_ATTN={_ATTN_OVERRIDES.get(AttentionBackendEnum.FLASH_ATTN)} "
            f"FLASHMLA_SPARSE={_ATTN_OVERRIDES.get(AttentionBackendEnum.FLASHMLA_SPARSE)}",
            flush=True,
        )


class MacaBackend(Backend):
    """
    METAX backend for operator implementations.

    This backend uses MACA libraries to provide high-performance
    operator implementations for METAX GPUs.
    """

    _available: bool | None = None

    @property
    def name(self) -> str:
        return "maca"

    @property
    def vendor(self) -> Optional[str]:
        return "metax"

    def is_available(self) -> bool:
        """Check if Metax hardware and libraries are available."""
        if MacaBackend._available is None:
            try:
                # Check if Metax device is available
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    MacaBackend._available = True
                else:
                    MacaBackend._available = False
            except Exception:
                MacaBackend._available = False
        return MacaBackend._available

    # ==================== Operator Implementations ====================

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU activation followed by element-wise multiplication.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import silu_and_mul_maca

        return silu_and_mul_maca(obj, x)

    def gelu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        GELU activation followed by element-wise multiplication.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import gelu_and_mul_maca

        return gelu_and_mul_maca(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        RMS normalization using Maca's CUDA implementation.
        """
        from .impl.layernorm import rms_norm_maca

        return rms_norm_maca(obj, x, residual)

    def rotary_embedding(
        self,
        obj,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_interleaved: bool = False,
        inplace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding using vLLM's CUDA implementation.
        """
        from .impl.rotary_embedding import rotary_embedding_maca

        return rotary_embedding_maca(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """
        Get the attention backend class path for CUDA.

        Supports:
        - FLASH_ATTN (default)
        - TRITON_ATTN (when use_flaggems_op("triton_attn") is True)
        - FLASHMLA_SPARSE (when use_mla and use_sparse are both True)

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)

        Returns:
            Fully qualified class path string
        """
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        # register before selection
        register_attention_backends()

        if use_mla:
            if use_sparse:
                return AttentionBackendEnum.FLASHMLA_SPARSE.get_path()
            return AttentionBackendEnum.FLASHMLA.get_path()

        # Default to FLASH_ATTN
        return AttentionBackendEnum.FLASH_ATTN.get_path()

    def topk_softmax(
        self,
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
    ):
        from .impl.fused_moe import topk_softmax_maca

        return topk_softmax_maca(
            topk_weights, topk_indices, token_expert_indices, gating_output, renormalize
        )

    def invoke_fused_moe_triton_kernel(
        self,
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        config,
        compute_type,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None,
        B_bias=None,
    ):
        from .impl.fused_moe import invoke_fused_moe_triton_kernel_maca

        invoke_fused_moe_triton_kernel_maca(
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=B_bias,
        )
