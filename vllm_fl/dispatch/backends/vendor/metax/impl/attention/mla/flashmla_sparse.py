# Copyright (c) 2026 BAAI. All rights reserved.

"""
GLM5.3-Flash sparse MLA backend for MetaX.

This is a thin subclass of vllm_metax's MacaFlashMLASparseBackend. It reuses the
vendor's kernels and impl unchanged; the ONLY behavioral change is forcing the
bf16 KV-cache path into "mixed_batch" mode.

Why (root cause):
  GLM5.3-Flash is a use-nope MLA: qk_rope_head_dim=0, so the MQA head_size is
  kv_lora_rank(512) + rope(0) = 512 -- NOT the 576 of DeepSeek V3.2 (512 NoPE +
  64 RoPE). The vendor's default bf16 "separate prefill/decode" path routes
  decode through flash_mla_with_kvcache -> flash_mla_cuda.fwd_kvcache_mla, whose
  bf16 sparse *decode* kernel HARD-asserts head_size == 576 and crashes on 512
  ("Expected head_size == 576 to be true, but got false").

  mixed_batch mode treats decode tokens as prefill and runs the whole batch
  through flash_mla_sparse_fwd (sparse_prefill_fwd), which is head_dim-parametric
  and already validated for GLM5.3's 512 layout in prefill. Slower per decode
  step, but correct -- day0 prioritizes correctness.

Why a subclass instead of editing vllm_metax:
  The vendor package lives in conda site-packages, is not version-controlled, and
  would need re-patching on every reinstall and on every machine. Subclassing
  keeps the change inside plugin-FL (git-tracked, pulled to both nodes) while the
  vendor's kernels/impl are reused verbatim.
"""

from __future__ import annotations

from vllm.logger import init_logger

from vllm_metax.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseMetadata,
    FlashMLASparseMetadataBuilder,
    MacaFlashMLASparseBackend,
)

logger = init_logger(__name__)


class Glm53FlashMLASparseMetadataBuilder(FlashMLASparseMetadataBuilder):
    """Builder that forces bf16 mixed_batch so decode avoids the 576-only kernel."""

    _mixed_batch_logged = False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
    ) -> FlashMLASparseMetadata:
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build=fast_build
        )
        # Only override for the bf16 KV-cache case (GLM5.3 runs bf16 cache). The
        # fp8 path has its own mixed_batch logic keyed on head count and is left
        # untouched.
        if self.use_bf16_kv_cache and not metadata.bf16_use_mixed_batch:
            metadata.bf16_use_mixed_batch = True
            if not Glm53FlashMLASparseMetadataBuilder._mixed_batch_logged:
                Glm53FlashMLASparseMetadataBuilder._mixed_batch_logged = True
                logger.info(
                    "[glm53-sparse-mla] forcing bf16_use_mixed_batch=True "
                    "(GLM5.3 use-nope MLA head_size=512; vendor bf16 decode "
                    "kernel is 576-only). Decode routes through the sparse "
                    "prefill kernel."
                )
        return metadata


class Glm53FlashMLASparseBackend(MacaFlashMLASparseBackend):
    """Vendor MetaX sparse MLA backend with a GLM5.3-specific metadata builder."""

    @staticmethod
    def get_name() -> str:
        # Keep the vendor's registry name so it slots into FLASHMLA_SPARSE.
        return "FLASHMLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type[Glm53FlashMLASparseMetadataBuilder]:
        return Glm53FlashMLASparseMetadataBuilder
