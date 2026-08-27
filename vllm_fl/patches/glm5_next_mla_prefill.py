# SPDX-License-Identifier: Apache-2.0
"""MLA prefill backend for out-of-tree (MetaX) platforms.

vLLM 0.24 split MLA into two independent code paths:

* ``MLAAttention.attn_backend`` — decode / main impl.  On MetaX this is
  dispatched by vllm-FL to the vendor FlashMLA / FlashMLASparse backend and
  works correctly.
* ``MLAAttention.prefill_backend`` — a *separate* prefill-only backend chosen
  unconditionally in ``MLAAttention.__init__`` via ``get_mla_prefill_backend``
  (``vllm/model_executor/layers/attention/mla_attention.py``).

The auto selector only offers ``FLASH_ATTN`` on non-Blackwell devices, and its
availability check probes ``vllm.vllm_flash_attn`` (the compiled
``_vllm_fa2_C`` extension).  MetaX does not ship that extension, so model
construction aborts with::

    ValueError: No valid MLA prefill backend found ...
                Reasons: {FLASH_ATTN: [required dependencies not available]}.

MetaX *does* provide a working standalone ``flash_attn`` wheel (the same one
vllm-metax uses for its MLA common path).  This module subclasses vLLM's
``FlashAttnPrefillBackend`` and rebinds it onto that standalone
``flash_attn_varlen_func``, then overrides the ``FLASH_ATTN`` prefill backend
registration so the auto selector resolves to a usable implementation.
"""

import logging

from vllm.v1.attention.backends.mla.prefill.flash_attn import (
    FlashAttnPrefillBackend,
)
from vllm.v1.attention.backends.mla.prefill.registry import (
    MLAPrefillBackendEnum,
    register_mla_prefill_backend,
)

logger = logging.getLogger(__name__)


class MacaFlashAttnPrefillBackend(FlashAttnPrefillBackend):
    """FlashAttention MLA prefill backed by the standalone ``flash_attn`` wheel.

    MetaX masquerades as a CUDA platform, but ``vllm.vllm_flash_attn`` (the
    ``_vllm_fa2_C`` extension) is unavailable.  We instead use the standalone
    ``flash_attn`` package.  Its ``flash_attn_varlen_func`` signature differs
    from vLLM's fork (``return_attn_probs`` rather than ``return_softmax_lse``
    / ``out`` / ``output_scale``), which is exactly the calling convention the
    parent class already implements for its non-vLLM (ROCm) branch.  We select
    that branch by forcing ``_is_vllm_fa = False``.
    """

    @classmethod
    def is_available(cls) -> bool:
        try:
            from flash_attn import flash_attn_varlen_func  # noqa: F401
        except ImportError:
            return False
        return True

    def __init__(self, *args, **kwargs) -> None:
        # The parent __init__ asserts a module-level flash_attn_varlen_func is
        # importable.  On MetaX that module-level symbol resolves to the vLLM
        # fork stub (None), so patch it in before delegating.
        import vllm.v1.attention.backends.mla.prefill.flash_attn as _fa_mod
        from flash_attn import flash_attn_varlen_func as _standalone_varlen

        _fa_mod.flash_attn_varlen_func = _standalone_varlen

        super().__init__(*args, **kwargs)

        # Force the standalone flash_attn code path regardless of what the
        # parent probed:
        #   * bind the standalone kernel,
        #   * no vLLM FA version (skip the fa_version functools.partial),
        #   * pad V to qk_head_dim (standalone FA needs equal head dims),
        #   * use the return_attn_probs calling convention (_is_vllm_fa=False).
        self.flash_attn_varlen_func = _standalone_varlen
        self.vllm_flash_attn_version = None
        self.requires_v_padding = True
        self._is_vllm_fa = False


def install_glm5_next_mla_prefill_backend() -> bool:
    """Override the FLASH_ATTN MLA prefill backend with the MetaX-compatible
    standalone flash_attn implementation.  Idempotent.

    Returns True if the override was installed, False if the standalone
    flash_attn wheel is unavailable (in which case vLLM's default selection is
    left untouched).
    """
    if not MacaFlashAttnPrefillBackend.is_available():
        logger.warning(
            "Standalone flash_attn wheel not importable; leaving vLLM's "
            "default MLA prefill backend selection in place. MLA prefill will "
            "fail on MetaX without it."
        )
        return False

    register_mla_prefill_backend(
        MLAPrefillBackendEnum.FLASH_ATTN,
        f"{MacaFlashAttnPrefillBackend.__module__}."
        f"{MacaFlashAttnPrefillBackend.__qualname__}",
    )
    logger.info(
        "Overrode FLASH_ATTN MLA prefill backend with standalone flash_attn "
        "implementation for out-of-tree platform"
    )
    return True


__all__ = [
    "MacaFlashAttnPrefillBackend",
    "install_glm5_next_mla_prefill_backend",
]
