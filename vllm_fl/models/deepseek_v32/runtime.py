# SPDX-License-Identifier: Apache-2.0
"""Model-scoped PPU BF16 DSA plan; leave other models and platforms unchanged."""

from functools import wraps
import os

import torch
from vllm.platforms import current_platform

from vllm_fl.activation import (
    ActivationPlan,
    MoEDispatchDefaults,
    PendingPatch,
    bind_patches,
    register_plan_provider,
)

_FINGERPRINT = (
    "deepseek_v32_ppu_bf16_indexer_piecewise64_norm_cacheprefix_admission48_v4@vllm0.24"
    if os.environ.get("VLLM_FL_PPU_BF16_PIECEWISE_GRAPH", "0") == "1"
    else "deepseek_v32_ppu_bf16_indexer_long64_eval_v1@vllm0.24"
)


def _apply():
    from vllm import _custom_ops
    from vllm.model_executor.models import deepseek_v2
    from vllm.model_executor.layers.attention import mla_attention
    from flaggems_vllm import concat_mla_q

    from .bf16_indexer import Bf16IndexerBackend, indexer_forward

    if os.environ.get("VLLM_FL_PPU_BF16_PIECEWISE_GRAPH", "0") == "1":
        # Register the public torch.library opaque metadata/cache boundary before
        # model tracing. Attention NEVER and communication exclusions stay active.
        from . import graph_boundary  # noqa: F401
        from vllm_fl.ops.layernorm import enable_ppu_graph_rms_norm

        enable_ppu_graph_rms_norm()

    original_cache = deepseek_v2.DeepseekV32IndexerCache

    class Bf16IndexerCache(original_cache):
        def __init__(self, head_dim, dtype, prefix, cache_config):
            if head_dim != 132 or dtype != torch.uint8:
                raise ValueError("Unsupported DeepSeek V3.2 indexer cache contract")
            super().__init__(128, torch.bfloat16, prefix, cache_config)

        def get_attn_backend(self):
            return Bf16IndexerBackend

    original_concat = _custom_ops.concat_mla_q

    @wraps(original_concat)
    def query_concat(q_nope, q_pe, q_out):
        return concat_mla_q(q_nope, q_pe, q_out)

    original_prefill = mla_attention.get_mla_prefill_backend

    @wraps(original_prefill)
    def prefill_backend(vllm_config):
        if vllm_config.attention_config.mla_prefill_backend is not None:
            return original_prefill(vllm_config)
        from .prefill import PpuMLAPrefillBackend

        return PpuMLAPrefillBackend

    entries = (
        (deepseek_v2, "DeepseekV32IndexerCache", Bf16IndexerCache),
        (deepseek_v2.Indexer, "forward", indexer_forward),
        (_custom_ops, "concat_mla_q", query_concat),
        (mla_attention, "get_mla_prefill_backend", prefill_backend),
    )
    bind_patches(
        [
            PendingPatch(
                target=f"{owner.__name__}.{attr}",
                owner=owner,
                attr=attr,
                replacement=replacement,
                pristine=getattr(owner, attr),
                fingerprint=_FINGERPRINT,
            )
            for owner, attr, replacement in entries
        ]
    )


def _attention(*, use_mla, use_sparse):
    if use_mla and use_sparse:
        return "vllm_fl.dispatch.backends.flaggems.impl.mla.FlagGemsSparseMLABackend"
    return None


def _provider(vllm_config):
    if os.environ.get("VLLM_FL_DEEPSEEK_V32_PPU_BF16", "0") != "1":
        return None
    model = getattr(vllm_config, "model_config", None)
    architectures = getattr(model, "architectures", ())
    if "DeepseekV32ForCausalLM" not in architectures:
        return None
    if getattr(current_platform, "vendor_name", None) != "thead":
        return None
    config = model.hf_config
    contract = (config.index_n_heads, config.index_head_dim, config.qk_rope_head_dim)
    if contract != (64, 128, 64) or model.dtype != torch.bfloat16:
        raise ValueError("PPU BF16 DSA pilot requires indexer H64/D128/RoPE64 and BF16")
    # Isolated long-context evaluation is opt-in, with independently validated
    # public long-sequence operators and the bounded8MiB indexer workspace.
    long_context = os.environ.get("VLLM_FL_PPU_BF16_LONG_CONTEXT_EVAL", "0") == "1"
    context_limit = 65536 if long_context else 8192
    if model.quantization is not None or not 1 <= model.max_model_len <= context_limit:
        raise ValueError(
            f"PPU BF16 DSA context requires unquantized1..{context_limit} tokens"
        )
    graph_enabled = os.environ.get("VLLM_FL_PPU_BF16_PIECEWISE_GRAPH", "0") == "1"
    long_admission_limit = 48 if graph_enabled else 32
    if (
        long_context
        and vllm_config.scheduler_config.max_num_seqs > long_admission_limit
    ):
        raise ValueError(
            f"PPU BF16 DSA long-context evaluation permits at most{long_admission_limit} sequences"
        )
    if vllm_config.speculative_config is not None:
        raise ValueError("PPU BF16 DSA pilot requires no speculation")
    if not model.enforce_eager:
        if not graph_enabled:
            raise ValueError(
                "PPU BF16 DSA non-eager execution requires explicit PIECEWISE opt-in"
            )
        compilation = vllm_config.compilation_config
        graph_mode = compilation.cudagraph_mode
        required_splits = {
            "vllm::unified_mla_attention_with_output",
            "vllm::unified_mla_kv_cache_update",
            "vllm::unified_kv_cache_update",
            "vllm_fl::ppu_bf16_indexer",
            "vllm::all_reduce",
            "vllm::all_gather",
            "vllm::reduce_scatter",
            "_c10d_functional::all_reduce",
            "_c10d_functional::all_gather_into_tensor",
            "_c10d_functional::reduce_scatter_tensor",
            "_c10d_functional::wait_tensor",
        }
        if (
            int(compilation.mode) != 3
            or compilation.backend != "eager"
            or graph_mode is None
            or not graph_mode.has_piecewise_cudagraphs()
            or graph_mode.has_full_cudagraphs()
            or not required_splits.issubset(set(compilation.splitting_ops or ()))
        ):
            raise ValueError(
                "PPU BF16 DSA graph requires VLLM_COMPILE/PIECEWISE with indexer and TP collectives excluded"
            )
    ops = ("moe_align_block_size", "invoke_fused_moe_triton_kernel")
    return ActivationPlan(
        name="deepseek_v32_ppu_bf16",
        fingerprint=_FINGERPRINT,
        apply=_apply,
        attention_backend=_attention,
        moe_defaults=MoEDispatchDefaults(
            whitelist_ops=ops,
            per_op_order=tuple((name, ("flagos",)) for name in ops),
            required_impls=tuple((name, ("flagos",)) for name in ops),
        ),
    )


def register_deepseek_v32_ppu_support():
    register_plan_provider(_provider)
