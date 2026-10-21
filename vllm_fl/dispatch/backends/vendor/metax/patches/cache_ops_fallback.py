# SPDX-License-Identifier: Apache-2.0
# 2026 - Added for MetaX C550 day-0 bring-up of MiniCPM-OCR-1.0.
"""Register ``_C_cache_ops::reshape_and_cache_flash`` for MetaX.

The MetaX image ships an *empty* vLLM build: the ``vllm._C`` / ``_C_cache_ops``
CUDA kernels are not compiled.  The MetaX FlashAttention backend, however, calls
``vllm._custom_ops.reshape_and_cache_flash`` directly for every dense-attention
layer, so the engine starts ("Application startup complete") and then dies on the
first request with::

    AttributeError: '_OpNamespace' '_C_cache_ops' object has no attribute
    'reshape_and_cache_flash'

Bind the op to FlagGems' Triton implementation (identical semantics, including
skipping negative slot mappings used for padding), with a pure-torch scatter as
fallback when FlagGems is unavailable.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_SCHEMA = (
    "reshape_and_cache_flash(Tensor key, Tensor value, Tensor! key_cache, "
    "Tensor! value_cache, Tensor slot_mapping, str kv_cache_dtype, "
    "Tensor k_scale, Tensor v_scale) -> ()"
)


def _reshape_and_cache_flash_torch(
    key: torch.Tensor,            # [num_tokens, num_kv_heads, head_size]
    value: torch.Tensor,          # [num_tokens, num_kv_heads, head_size]
    key_cache: torch.Tensor,      # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache: torch.Tensor,    # same as key_cache
    slot_mapping: torch.Tensor,   # [num_tokens] int64, negative = skip
    kv_cache_dtype: str,          # unused for non-quantized caches
    k_scale: torch.Tensor,        # unused for non-quantized caches
    v_scale: torch.Tensor,        # unused for non-quantized caches
) -> None:
    """Pure-torch scatter of per-token K/V into the paged cache."""
    _, block_size, num_kv_heads, head_size = key_cache.shape
    key = key.reshape(-1, num_kv_heads, head_size)
    value = value.reshape(-1, num_kv_heads, head_size)

    # The op is sized by slot_mapping, not by the (padded) key/value tensors.
    slots = slot_mapping.flatten()
    num_tokens = slots.numel()
    key = key[:num_tokens].to(key_cache.dtype)
    value = value[:num_tokens].to(value_cache.dtype)

    valid = (slots >= 0).view(-1, 1, 1)
    slots = torch.clamp(slots, min=0)
    block_idx = slots // block_size
    block_offset = slots % block_size

    key_cache[block_idx, block_offset] = torch.where(
        valid, key, key_cache[block_idx, block_offset]
    )
    value_cache[block_idx, block_offset] = torch.where(
        valid, value, value_cache[block_idx, block_offset]
    )


def _resolve_impl():
    try:
        import flag_gems

        impl = getattr(flag_gems, "reshape_and_cache_flash", None)
        if impl is not None:
            return impl
        logger.warning("flag_gems has no reshape_and_cache_flash; using torch fallback")
    except Exception as exc:  # pragma: no cover - depends on the image
        logger.warning("FlagGems unavailable (%s); using torch fallback", exc)
    return _reshape_and_cache_flash_torch


# The Library object owns the registration: torch unregisters the schema in
# Library.__del__, so it must stay alive for the lifetime of the process.
_LIB: torch.library.Library | None = None


def register_reshape_and_cache_flash() -> None:
    global _LIB
    """Bind ``_C_cache_ops::reshape_and_cache_flash`` if the kernel is missing."""
    if getattr(register_reshape_and_cache_flash, "_done", False):
        return
    register_reshape_and_cache_flash._done = True

    if hasattr(torch.ops._C_cache_ops, "reshape_and_cache_flash"):
        return  # native kernel present, nothing to do

    lib = torch.library.Library("_C_cache_ops", "FRAGMENT")
    _LIB = lib
    try:
        lib.define(_SCHEMA)
    except Exception as exc:
        logger.warning("Failed to define _C_cache_ops::reshape_and_cache_flash: %s", exc)
        return
    lib.impl("reshape_and_cache_flash", _resolve_impl(), "CompositeExplicitAutograd")
    logger.info(
        "Registered _C_cache_ops::reshape_and_cache_flash fallback for MetaX"
    )


register_reshape_and_cache_flash()
