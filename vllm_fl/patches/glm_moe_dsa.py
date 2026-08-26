# SPDX-License-Identifier: Apache-2.0
"""GLM-5 (GlmMoeDsa) specific patches for vLLM 0.13.0 compatibility.

All monkey-patches required to run GLM-5 FP8 on the current environment
(transformers 4.57.6, CUDA 13.1, no deep_gemm JIT) are collected here.
"""

import logging

logger = logging.getLogger(__name__)


def patch_tokenizer_compat():
    """Patch transformers tokenizer loading for 5.x compat on 4.57.6.

    GLM-5's tokenizer uses transformers 5.x naming (TokenizersBackend) and
    special_tokens format (list instead of dict). This patches both issues
    so the tokenizer loads correctly on transformers 4.57.6.
    """
    try:
        import transformers.models.auto.tokenization_auto as ta

        if not getattr(ta, "_fl_patched", False):
            _orig = ta.tokenizer_class_from_name

            def _patched(class_name):
                result = _orig(class_name)
                if result is None and "TokenizersBackend" in class_name:
                    from transformers import PreTrainedTokenizerFast
                    return PreTrainedTokenizerFast
                return result

            ta.tokenizer_class_from_name = _patched
            ta._fl_patched = True
    except Exception:
        pass

    try:
        import transformers.tokenization_utils_base as tub

        if not getattr(tub.SpecialTokensMixin, "_fl_patched_special", False):
            _orig_set = tub.SpecialTokensMixin._set_model_specific_special_tokens

            def _patched_set(self, special_tokens=None):
                if isinstance(special_tokens, list):
                    special_tokens = {t: t for t in special_tokens}
                return _orig_set(self, special_tokens=special_tokens)

            tub.SpecialTokensMixin._set_model_specific_special_tokens = _patched_set
            tub.SpecialTokensMixin._fl_patched_special = True
    except Exception:
        pass


def patch_indexer_schedule_metadata():
    """Fill decode ``schedule_metadata`` on non-CUDA (MetaX) accelerators.

    Root cause (confirmed by runtime dumps, hy4-metax log rounds 12-13):
    upstream ``DeepseekV32IndexerMetadataBuilder.build`` gates the
    ``schedule_metadata`` computation behind
    ``if current_platform.is_cuda() and has_deep_gemm():`` (indexer.py:616),
    yet it *unconditionally* stores the buffer into the decode metadata
    (indexer.py:625). On MetaX ``current_platform.is_cuda()`` is False (it is
    cuda_alike but not NVIDIA), so the buffer is left as the uninitialised
    ``torch.empty((num_sms+1, 2), int32)`` allocation. The bf16 paged MQA
    logits kernel then reads that garbage as its SM schedule and walks off
    into unmapped memory -> MACA Xnack/ATU Fault (== CUDA illegal memory
    access). Dumps showed schedule_metadata full of billion-scale +/- ints
    while the correct table is small 0/1 values.

    Fix: wrap ``build`` and, whenever a decode metadata was produced, recompute
    ``schedule_metadata`` ourselves — regardless of ``is_cuda()``. We reuse the
    *already-correct* ``result.decode.seq_lens`` that ``build`` just computed
    (it went through the compress_ratio conversion + 2D unsqueeze), rather than
    re-deriving from the raw ``common_attn_metadata.seq_lens``. This keeps us in
    lock-step with upstream's own arguments and avoids parallel-logic drift.
    We also use ``storage_block_size`` (with ``block_size`` fallback) to match
    upstream indexer.py:619.
    """
    from vllm.utils.import_utils import has_deep_gemm
    if not has_deep_gemm():
        logger.warning("[hy4-sched-meta] has_deep_gemm() is False; paged MQA "
                       "kernel unused, skipping schedule_metadata patch")
        return

    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )
    from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata

    if getattr(DeepseekV32IndexerMetadataBuilder, "_fl_sched_meta_patched", False):
        return
    _orig_build = DeepseekV32IndexerMetadataBuilder.build

    def _patched_build(self, common_prefix_len, common_attn_metadata,
                       fast_build=False):
        result = _orig_build(self, common_prefix_len,
                             common_attn_metadata, fast_build)
        decode = getattr(result, "decode", None)
        if decode is not None and decode.schedule_metadata is not None:
            # Reuse the seq_lens build() already stored on the decode metadata
            # (compress_ratio-adjusted, 2D). This is exactly what the kernel
            # will index against, so the schedule must be derived from it.
            block_size = getattr(self.kv_cache_spec, "storage_block_size", None)
            if block_size is None:
                block_size = self.kv_cache_spec.block_size
            self.scheduler_metadata_buffer[:] = get_paged_mqa_logits_metadata(
                decode.seq_lens, block_size, self.num_sms
            )
            if not getattr(_patched_build, "_logged", False):
                _patched_build._logged = True
                logger.warning(
                    "[hy4-sched-meta] recomputed schedule_metadata on non-CUDA "
                    "path (block_size=%s num_sms=%s seq_lens.shape=%s) -> buffer "
                    "now initialised, decode paged kernel safe",
                    block_size, self.num_sms, tuple(decode.seq_lens.shape),
                )
        return result

    DeepseekV32IndexerMetadataBuilder.build = _patched_build
    DeepseekV32IndexerMetadataBuilder._fl_sched_meta_patched = True
    logger.warning("[hy4-sched-meta] patched DeepseekV32IndexerMetadataBuilder."
                   "build: schedule_metadata always computed when deep_gemm "
                   "available (fixes MetaX is_cuda() gate leaving it garbage)")


def apply_platform_patches():
    """All GLM-5 patches needed at platform registration time.

    Runs in ``register()`` in every process (incl. ray workers), so the
    indexer schedule_metadata fix lands in the workers that actually build
    decode metadata — not just the driver.
    """
    patch_tokenizer_compat()
    try:
        patch_indexer_schedule_metadata()
    except Exception as e:
        logger.warning("[hy4-sched-meta] failed to apply schedule_metadata "
                       "patch: %s", e)

def patch_indexer_rope_reshape():
    """Fix RoPE output shape in Indexer.forward for DSA models.

    vLLM 0.13.0 uses squeeze(0) / squeeze((0, 2)) on RoPE outputs, which
    can fail when the RoPE implementation introduces extra leading dims.
    Replace squeeze with explicit reshape for robustness.
    """
    import torch
    from vllm.model_executor.models.deepseek_v2 import (
        Indexer,
        per_token_group_quant_fp8,
    )

    def _patched_forward(self, hidden_states, qr, positions, rotary_emb):
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        k, _ = self.wk(hidden_states)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # Use reshape instead of squeeze to handle extra leading dims
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)

        # quant q (k quant is fused with cache insertion)
        q = q.view(-1, self.head_dim)
        q_fp8, q_scale = per_token_group_quant_fp8(
            q,
            self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=self.scale_fmt is not None,
        )
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)

        weights, _ = self.weights_proj(hidden_states)
        weights = (
            weights.unsqueeze(-1) * q_scale * self.softmax_scale
            * self.n_head**-0.5
        )
        weights = weights.squeeze(-1)

        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache[0],
            q_fp8,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

    Indexer.forward = _patched_forward
    logger.info("Patched Indexer.forward: reshape RoPE outputs to ensure "
                "correct dims")


def apply_model_patches():
    """All GLM-5 patches needed at model registration time."""
    patch_indexer_schedule_metadata()
    patch_indexer_rope_reshape()
