# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MetaX (MACA) implementation of FlashMLA Sparse backend.

Overrides vLLM's native FLASHMLA_SPARSE which requires vllm._flashmla_C
(NVIDIA-only). Routes sparse MLA decode to MetaX's flash_mla library instead.
"""

import os
from typing import Any, ClassVar

import torch

from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
    FlashMLASparseMetadata,
    FlashMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)

from ..ops.flashmla import (
    flash_mla_sparse_prefill,
    is_flashmla_sparse_supported,
    torch_flash_mla_sparse_prefill,
)

# Round 22 diagnostic: swap the MetaX C sparse-prefill kernel for the known-good
# pure-torch reference (torch_flash_mla_sparse_prefill, which correctly masks
# invalid/-1 indices to -inf and applies sm_scale*log2(e)). Set to 1 to test
# whether the "repeat last token" garbage lives in the MetaX kernel's numerics /
# invalid-index masking. Default off => original behaviour. Reversible.
_HY4_SPARSE_TORCH_REF = os.environ.get("VLLM_HY4_SPARSE_TORCH_REF", "0") == "1"

logger = init_logger(__name__)


@register_backend(AttentionBackendEnum.FLASHMLA_SPARSE)
class MacaFlashMLASparseBackend(FlashMLASparseBackend):
    """MetaX FLASHMLA_SPARSE backend using MACA flash_mla library."""

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        supported, _ = is_flashmla_sparse_supported()
        return supported

    @staticmethod
    def get_builder_cls() -> type[FlashMLASparseMetadataBuilder]:
        return FlashMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["MacaFlashMLASparseImpl"]:
        return MacaFlashMLASparseImpl


class MacaFlashMLASparseImpl(FlashMLASparseImpl):
    """MetaX impl that calls flash_mla_sparse_prefill for BF16 sparse MLA."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # MetaX doesn't have the hopper/blackwell head padding constraint
        self.prefill_padding = 64

    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        # MetaX flash_mla requires num_heads to be a multiple of 64
        needs_padding = self.num_heads % self.prefill_padding != 0
        if needs_padding:
            assert self.prefill_padding % self.num_heads == 0
            logger.warning_once(
                f"Padding num_heads from {self.num_heads} to "
                f"{self.prefill_padding} for MetaX BF16 sparse kernel"
            )
            q_padded = q.new_zeros(
                (q.shape[0], self.prefill_padding, q.shape[2])
            )
            q_padded[:, : self.num_heads, :] = q
            q = q_padded

        topk_indices = topk_indices.view(num_tokens, 1, -1)
        if _HY4_SPARSE_TORCH_REF:
            # Diagnostic path: known-good torch ref that masks invalid indices.
            logger.warning_once(
                "[hy4-sparse] VLLM_HY4_SPARSE_TORCH_REF=1 -> using "
                "torch_flash_mla_sparse_prefill (masks -1) instead of MetaX "
                "flash_mla_sparse_prefill. DIAGNOSTIC ONLY, slow."
            )
            output, _max_logits, _lse = torch_flash_mla_sparse_prefill(
                q,
                kv_c_and_k_pe_cache,
                topk_indices,
                self.softmax_scale,
            )
        else:
            output, _max_logits, _lse = flash_mla_sparse_prefill(
                q,
                kv_c_and_k_pe_cache,
                topk_indices,
                self.softmax_scale,
            )

        output = output[:, : self.num_heads, :]
        return output

    def _fp8_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        kernel_metadata: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "FP8 sparse MLA decode is not supported on MetaX. "
            "Use kv_cache_dtype='auto' (bf16) instead."
        )
