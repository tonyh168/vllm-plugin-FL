# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only HY4 model for vLLM serving."""

from __future__ import annotations

import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE, GateLinear
from vllm.model_executor.layers.fused_moe.layer import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_v2 import _try_load_fp8_indexer_wk
from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    get_pp_missing_layer_names,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_fl.configs.hy_v4 import HYV4Config
from vllm_fl.models.hy_v4_attention import (
    HYV4MLAAttention,
    compute_skip_topk_layers,
    is_skip_topk_indexer_weight,
)

logger = init_logger(__name__)


def _try_load_mxfp8_indexer_wk(
    name: str,
    tensor: torch.Tensor,
    pending: dict[str, dict[str, torch.Tensor]],
    params_dict: dict[str, nn.Parameter],
    loaded_params: set[str],
    pp_missing_layer_names: Iterable[str] = (),
) -> bool:
    """Fuse this checkpoint's ModelOpt MXFP8 indexer wk with weights_proj."""
    if ".indexer.wk." not in name:
        return False
    is_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
    is_scale = name.endswith(".weight_scale") and tensor.dtype == torch.uint8
    if not is_weight and not is_scale:
        return False

    # DIAGNOSTIC (2026-08-30): thead's migration has no MXFP8 indexer-wk branch,
    # so on a BF16/FP8 checkpoint this path should never fire. Log once the first
    # time a genuine MXFP8 indexer.wk tensor is seen so we can confirm from the
    # load-time logs whether this checkpoint actually exercises the MXFP8 fusion.
    # If it turns out this must never trigger, replace this with an assert.
    logger.warning_once(
        "[hy4-mxfp8-indexer] MXFP8 indexer.wk tensor detected (name=%s, "
        "dtype=%s) -> _try_load_mxfp8_indexer_wk fusion path IS active. If "
        "this checkpoint is expected to be BF16/FP8 (like thead), this branch "
        "should not run.",
        name,
        tensor.dtype,
    )

    layer_prefix = name.rsplit(".wk.", 1)[0]
    if any(name.startswith(prefix) for prefix in pp_missing_layer_names):
        return True
    entry = pending.setdefault(layer_prefix, {})
    entry["weight" if is_weight else "scale"] = tensor
    if "weight" not in entry or "scale" not in entry:
        return True

    weight, scale = entry["weight"], entry["scale"]
    del pending[layer_prefix]
    if weight.shape[:-1] != scale.shape[:-1]:
        raise ValueError(
            f"HY4 indexer MXFP8 shape mismatch: {weight.shape} vs {scale.shape}"
        )
    group_size = weight.shape[-1] // scale.shape[-1]
    if group_size != 32:
        raise ValueError(
            f"HY4 indexer MXFP8 expected group size 32, got {group_size}"
        )
    scales = torch.exp2(scale.to(torch.int16).float() - 127.0)
    scales = scales.repeat_interleave(group_size, dim=-1)
    weight_bf16 = (weight.float() * scales).to(torch.bfloat16)

    fused_name = f"{layer_prefix}.wk_weights_proj.weight"
    param = params_dict[fused_name]
    param.weight_loader(param, weight_bf16, 0)
    loaded_params.add(fused_name)
    return True


class HYV4HyperConnection(nn.Module):
    """Four identity residual streams with token-dependent read/write gates."""

    def __init__(self, config: HYV4Config) -> None:
        super().__init__()
        self.num_streams = config.hc_mult
        width = config.hc_mult * config.hidden_size
        self.hc_fn = nn.Parameter(
            torch.empty(2 * config.hc_mult, width, dtype=torch.float32)
        )
        self.hc_base = nn.Parameter(
            torch.empty(2 * config.hc_mult, dtype=torch.float32)
        )
        self.hc_scale = nn.Parameter(torch.empty(2, dtype=torch.float32))
        self.magnitude = float(config.hc_magnitude)
        self.hc_eps = float(config.hc_eps)
        self.normalize_eps = float(config.rms_norm_eps)

    def read(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapse streams and return the write gates for the branch output."""
        flat = streams.flatten(1).float()
        norm = torch.rsqrt(
            flat.square().mean(dim=-1, keepdim=True) + self.normalize_eps
        )
        gates = F.linear(flat, self.hc_fn) * norm
        read, write = gates.split(self.num_streams, dim=-1)
        read = torch.sigmoid(read * self.hc_scale[0] + self.hc_base[: self.num_streams])
        write = self.magnitude * torch.sigmoid(
            write * self.hc_scale[1] + self.hc_base[self.num_streams :]
        )
        read = read + self.hc_eps
        write = write + self.hc_eps
        collapsed = (read.unsqueeze(-1) * streams.float()).sum(dim=1)
        return collapsed.to(streams.dtype), write

    @staticmethod
    def write(
        streams: torch.Tensor,
        delta: torch.Tensor,
        write: torch.Tensor,
    ) -> torch.Tensor:
        """Write one branch result into all identity residual streams."""
        dtype = delta.dtype
        return (
            streams.float()
            + write.float().unsqueeze(-1) * delta.float().unsqueeze(1)
        ).to(dtype)


class HYV4HyperLayer(nn.Module):
    """Checkpoint-compatible wrapper around a pre-branch hyper-connection."""

    def __init__(self, config: HYV4Config) -> None:
        super().__init__()
        self.hc_pre = HYV4HyperConnection(config)


class HYV4HyperHead(nn.Module):
    """Collapse the four final residual streams before the final RMSNorm."""

    def __init__(self, config: HYV4Config) -> None:
        super().__init__()
        self.num_streams = config.hc_mult
        width = config.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(config.hc_mult, width, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(config.hc_mult, dtype=torch.float32)
        )
        self.hc_head_scale = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.hc_eps = float(config.hc_eps)
        self.normalize_eps = float(config.rms_norm_eps)

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        flat = streams.flatten(1).float()
        norm = torch.rsqrt(
            flat.square().mean(dim=-1, keepdim=True) + self.normalize_eps
        )
        read = torch.sigmoid(
            F.linear(flat, self.hc_head_fn) * norm * self.hc_head_scale
            + self.hc_head_base
        ) + self.hc_eps
        hidden = (read.unsqueeze(-1) * streams.float()).sum(dim=1)
        return hidden.to(streams.dtype)


class HYV4DenseMLP(nn.Module):
    """Tensor-parallel unclamped SwiGLU used by dense and shared branches."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class HYV4MoE(nn.Module):
    """HY4 no-aux sigmoid routed experts plus one shared expert."""

    def __init__(
        self,
        config: HYV4Config,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config
        if parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "HY4 does not yet support sequence-parallel MoE; disable it."
            )
        if config.hidden_act != "silu":
            raise ValueError("HY4 currently supports only the silu activation")

        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        # The SM90 DSV3 router kernel selected by vLLM for H=6144/E=256
        # requires BF16 weights. HY4 stores and evaluates the router in FP32,
        # so keep this model on GateLinear's FP32 F.linear fallback instead.
        self.gate.allow_dsv3_router_gemm = False
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32)
        )
        self.shared_experts = HYV4DenseMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.shared_experts",
        )

        eplb_config = parallel_config.eplb_config
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = config.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts

        # Keep the shared branch outside FusedMoE so HY4's FP32 routed/shared
        # residual addition is explicit and cannot be rounded early to BF16.
        self.experts = FusedMoE(
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            routed_scaling_factor=config.routed_scaling_factor,
            swiglu_limit=config.swiglu_limit,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            apply_routed_scale_to_output=False,
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            router_logits_dtype=torch.float32,
        )
        self.n_local_physical_experts = self.experts.routed_experts.local_num_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, shape[-1])
        if self.experts.is_internal_router:
            routed = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            routed = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        shared = self.shared_experts(hidden_states)
        combined = (routed.float() + shared.float()).to(hidden_states.dtype)
        return combined.view(shape)


class HYV4Attention(HYV4MLAAttention):
    """HY4 compressed MLA plus DSA indexer, gate, and learnable sink."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: HYV4Config,
        layer_idx: int,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
            layer_idx=layer_idx,
        )


class HYV4DecoderLayer(nn.Module):
    """One HY4 attention/MLP block operating on four residual streams."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None,
    ) -> None:
        super().__init__()
        config = typing.cast(HYV4Config, vllm_config.model_config.hf_config)
        layer_idx = int(prefix.rsplit(".", 1)[-1])
        self.hc_attn_layer = HYV4HyperLayer(config)
        self.self_attn = HYV4Attention(
            vllm_config,
            config,
            layer_idx,
            f"{prefix}.self_attn",
            topk_indices_buffer,
        )
        self.hc_mlp_layer = HYV4HyperLayer(config)
        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = HYV4MoE(config, vllm_config, f"{prefix}.mlp")
        else:
            self.mlp = HYV4DenseMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        streams: torch.Tensor,
    ) -> torch.Tensor:
        hidden, write = self.hc_attn_layer.hc_pre.read(streams)
        hidden = self.input_layernorm(hidden)
        hidden = self.self_attn(positions, hidden)
        streams = self.hc_attn_layer.hc_pre.write(streams, hidden, write)

        hidden, write = self.hc_mlp_layer.hc_pre.read(streams)
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        streams = self.hc_mlp_layer.hc_pre.write(streams, hidden, write)
        return streams


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class HYV4Model(nn.Module):
    """HY4 decoder backbone."""

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = typing.cast(HYV4Config, vllm_config.model_config.hf_config)
        self.config = config
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=vllm_config.device_config.device,
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: HYV4DecoderLayer(
                vllm_config,
                prefix,
                self.topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.hc_head = HYV4HyperHead(config)
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.hc_head = PPMissingLayer()
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["streams"],
            config.hc_mult * config.hidden_size,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                inputs_embeds = self.embed_input_ids(input_ids)
            streams = inputs_embeds.unsqueeze(1).expand(
                -1,
                self.config.hc_mult,
                -1,
            )
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline rank requires intermediate tensors")
            streams = intermediate_tensors["streams"].view(
                -1,
                self.config.hc_mult,
                self.config.hidden_size,
            )

        # Round 24b: trace the multi-stream residual across layers. L0 attention
        # was verified healthy, so the garbage is either accumulating across
        # layers or in the hc_head merge / final norm. Only on real steps, capped.
        # DEBUG DUMPS DISABLED (2026-08-30): the [hy4-fwd] per-layer stream dumps
        # and [hy4-phase1-hidden] cosine probe below flooded the server log with
        # intermediate tensor stats. Hard-gated off; flip _dbg back to the block
        # below only for targeted debugging.
        _dbg = False
        # try:
        #     from vllm.forward_context import get_forward_context
        #     _dbg = (
        #         isinstance(getattr(get_forward_context(), "attn_metadata", None), dict)
        #         and getattr(HYV4Model, "_hy4_fwd_dbg_n", 0) < 6
        #     )
        # except Exception:
        #     _dbg = False
        if _dbg:
            HYV4Model._hy4_fwd_dbg_n = getattr(HYV4Model, "_hy4_fwd_dbg_n", 0) + 1

        def _dbg_stream(tag, s):
            if not _dbg:
                return
            try:
                f = s.detach().float().reshape(-1)
                lt = s.detach().float().reshape(s.shape[0], -1)[-1].norm().item()
                logger.warning(
                    "[hy4-fwd] %s shape=%s norm=%.4f mean=%.4e std=%.4e "
                    "min=%.4e max=%.4e nan=%s inf=%s last_tok_norm=%.4f",
                    tag, tuple(s.shape), float(f.norm()), float(f.mean()),
                    float(f.std()), float(f.min()), float(f.max()),
                    bool(f.isnan().any()), bool(f.isinf().any()), float(lt),
                )
            except Exception as e:
                logger.warning("[hy4-fwd] %s failed: %s", tag, e)

        _dbg_stream(f"streams.in (start_layer={self.start_layer})", streams)
        for _i, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
            streams = layer(positions, streams)
            if _dbg and (self.start_layer + _i) in (0, 1, self.end_layer - 1):
                _dbg_stream(f"streams after layer {self.start_layer + _i}", streams)

        if not get_pp_group().is_last_rank:
            _dbg_stream("streams.out (-> next PP)", streams)
            return IntermediateTensors({"streams": streams.flatten(1)})
        hidden_states = self.hc_head(streams)
        _dbg_stream("hc_head out", hidden_states)
        normed = self.norm(hidden_states)
        _dbg_stream("final norm out", normed)
        # Phase 1 (new_repeat_plan): decide "attention family" vs "downstream
        # family". If the repeat-last-token garbage is a HIDDEN collapse, the
        # per-position final-hidden vectors are nearly identical (pairwise
        # cosine ~1, tiny cross-position variance) -> bug is upstream in
        # attention/MoE. If positions are well-separated here, hidden is fine
        # and the collapse must be in lm_head/sampler (Phase 4b). Self-contained
        # invariant, no golden needed. Gated to real steps only.
        if _dbg:
            try:
                h = normed.detach().float()
                h2d = h.reshape(h.shape[0], -1)  # [num_tokens, hidden]
                n_tok = h2d.shape[0]
                if n_tok >= 2:
                    hn = torch.nn.functional.normalize(h2d, dim=-1)
                    # full pairwise cosine matrix (small n_tok during prefill)
                    cos = hn @ hn.t()
                    off = cos - torch.eye(n_tok, device=cos.device)
                    # off-diagonal stats: mean/max/min cosine between DISTINCT
                    # positions. mean≈1 => collapse.
                    denom = max(1, n_tok * n_tok - n_tok)
                    off_mean = off.sum().item() / denom
                    triu = cos[torch.triu(torch.ones_like(cos), diagonal=1) > 0]
                    # cross-position variance of the hidden itself (per-dim std
                    # averaged): ~0 => all positions identical.
                    cross_std = h2d.std(dim=0).mean().item()
                    logger.warning(
                        "[hy4-phase1-hidden] n_tok=%s | pairwise_cos off-diag "
                        "mean=%.4f max=%.4f min=%.4f | cross_pos_std=%.4e | "
                        "per_tok_norm first=%.3f mid=%.3f last=%.3f | "
                        "COLLAPSE(cos~1)=%s",
                        n_tok, off_mean,
                        float(triu.max()) if triu.numel() else float("nan"),
                        float(triu.min()) if triu.numel() else float("nan"),
                        cross_std,
                        float(h2d[0].norm()),
                        float(h2d[n_tok // 2].norm()),
                        float(h2d[-1].norm()),
                        bool(off_mean > 0.99),
                    )
                else:
                    logger.warning(
                        "[hy4-phase1-hidden] n_tok=%s (single token, no "
                        "pairwise cosine); per_tok_norm=%.3f",
                        n_tok, float(h2d[0].norm()),
                    )
            except Exception as _e:
                logger.warning("[hy4-phase1-hidden] dump failed: %s", _e)
        return normed


class HYV4LogitsProcessor(LogitsProcessor):
    """Compute the excluded HY4 language head in FP32 when configured."""

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        hidden_states = hidden_states.to(lm_head.weight.dtype)
        logits = lm_head.quant_method.apply(
            lm_head,
            hidden_states,
            bias=embedding_bias,
        )
        logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        # Round 24b: dump the last-token logits argmax to see if lm_head maps the
        # final hidden to the repeated token (garbage) vs a sane distribution.
        try:
            # Gate on real steps only: without this, vLLM's _dummy_run/profiling
            # passes (attn_metadata is NOT a dict) consume the debug budget and
            # report bogus n_tok (e.g. 256). Match [hy4-phase1-hidden] gating so
            # prefill (n_tok=4) then decode steps (n_tok=1) are captured, which is
            # where the repeat actually manifests.
            # DEBUG DUMPS DISABLED (2026-08-30): [hy4-logits]/[hy4-phase1-logits]
            # argmax/top-k probes are diagnostic-only. Hard-gated off; restore the
            # block below for targeted debugging.
            _real_step = False
            # try:
            #     from vllm.forward_context import get_forward_context
            #     _real_step = isinstance(
            #         getattr(get_forward_context(), "attn_metadata", None), dict)
            # except Exception:
            #     _real_step = False
            if (_real_step
                    and getattr(HYV4ForCausalLM, "_hy4_logit_dbg_n", 0) < 6
                    and logits is not None):
                HYV4ForCausalLM._hy4_logit_dbg_n = getattr(
                    HYV4ForCausalLM, "_hy4_logit_dbg_n", 0) + 1
                lg = logits.detach().float()
                last = lg[-1]
                top = last.topk(min(5, last.numel()))
                logger.warning(
                    "[hy4-logits] in_hidden_norm=%.4f logits shape=%s | "
                    "last_tok top5 ids=%s vals=%s | logit min=%.3f max=%.3f "
                    "nan=%s inf=%s",
                    float(hidden_states.detach().float().reshape(
                        hidden_states.shape[0], -1)[-1].norm()),
                    tuple(logits.shape), top.indices.tolist(),
                    [round(v, 3) for v in top.values.tolist()],
                    float(last.min()), float(last.max()),
                    bool(lg.isnan().any()), bool(lg.isinf().any()),
                )
                # Phase 1 (new_repeat_plan): per-position argmax. If EVERY
                # position's argmax is the same token id, the logits have
                # collapsed regardless of hidden health -> confirms a downstream
                # (lm_head/sampler) collapse when paired with a healthy
                # [hy4-phase1-hidden]. If argmax varies sanely per position but
                # output still repeats, the collapse is in sampling/detokenizer.
                if lg.dim() == 2 and lg.shape[0] >= 2:
                    argmax = lg.argmax(dim=-1)  # [num_tokens]
                    n_tok = argmax.shape[0]
                    uniq = torch.unique(argmax)
                    logger.warning(
                        "[hy4-phase1-logits] PREFILL n_tok=%s | per_pos_argmax "
                        "first8=%s last=%s | num_unique_argmax=%s | "
                        "ALL_SAME(collapse)=%s",
                        n_tok, argmax[:8].tolist(), int(argmax[-1].item()),
                        int(uniq.numel()), bool(uniq.numel() == 1),
                    )
                elif lg.dim() == 2 and lg.shape[0] == 1:
                    # DECODE step: single query token. The repeat manifests here
                    # -- log the winning id + margin over runner-up each step so
                    # we can watch the `=?=?` sequence form at the logits level
                    # and see whether the margin is a confident lock or a
                    # near-tie (numerical) collapse.
                    row = lg[0]
                    t2 = row.topk(min(2, row.numel()))
                    win = int(t2.indices[0].item())
                    margin = (float(t2.values[0]) - float(t2.values[1])
                              if t2.values.numel() > 1 else float("nan"))
                    hist = getattr(HYV4ForCausalLM, "_hy4_decode_ids", [])
                    hist.append(win)
                    HYV4ForCausalLM._hy4_decode_ids = hist
                    logger.warning(
                        "[hy4-phase1-logits] DECODE step=%s | win_id=%s "
                        "margin=%.4f top2_ids=%s top2_vals=%s | seq_so_far=%s",
                        len(hist), win, margin, t2.indices.tolist(),
                        [round(v, 3) for v in t2.values.tolist()], hist[-8:],
                    )
        except Exception as _e:
            logger.warning("[hy4-logits] dump failed: %s", _e)
        return logits


class HYV4MixtureOfExperts(MixtureOfExperts):
    """Expose HY4 expert metadata to vLLM EPLB infrastructure."""

    moe_mlp_layers: list[HYV4MoE]

    def extract_moe_parameters(self, example_moe: HYV4MoE | None) -> None:
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            return
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        if self.num_local_physical_experts != num_local_physical_experts:
            raise ValueError("HY4 local expert count changed unexpectedly")
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.routed_experts.update_expert_map()


class HYV4ForCausalLM(
    nn.Module,
    SupportsPP,
    HYV4MixtureOfExperts,
):
    """Native vLLM causal language model for the HY4 preview checkpoint."""

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = typing.cast(HYV4Config, vllm_config.model_config.hf_config)
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.model = HYV4Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if get_pp_group().is_last_rank:
            lm_head_dtype = (
                torch.float32
                if config.enable_lm_head_fp32
                else vllm_config.model_config.dtype
            )
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                params_dtype=lm_head_dtype,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = HYV4LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.expert_weights = []
        self.num_expert_groups = 1
        self.moe_mlp_layers = []
        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, HYV4MoE):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts.routed_experts)
        self.num_moe_layers = len(self.moe_layers)
        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    @staticmethod
    def _packed_expert_target(name: str) -> tuple[str, bool] | None:
        gate_up_name = ".mlp.experts.gate_up_proj"
        down_name = ".mlp.experts.down_proj"
        if gate_up_name in name:
            return (
                name.replace(
                    gate_up_name,
                    ".mlp.experts.routed_experts.w13_weight",
                ),
                True,
            )
        if down_name in name:
            return (
                name.replace(
                    down_name,
                    ".mlp.experts.routed_experts.w2_weight",
                ),
                False,
            )
        return None

    def _load_all_experts(
        self,
        name: str,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        loader = typing.cast(Callable[..., bool], param.weight_loader)
        loaded_any = False
        expert_ids = getattr(self, "_hy4_local_expert_ids", range(num_experts))
        if loaded_weight.shape[0] != len(expert_ids):
            raise ValueError(
                f"HY4 packed expert leading dimension {loaded_weight.shape[0]} "
                f"does not match {len(expert_ids)} selected experts for {name}"
            )
        for loaded_expert_id, expert_id in enumerate(expert_ids):
            loaded_any |= loader(
                param,
                loaded_weight[loaded_expert_id],
                name,
                shard_id=shard_id,
                expert_id=expert_id,
                return_success=True,
            )
        return loaded_any

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load HY4, including its checkpoint-packed ModelOpt MXFP8 experts."""
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        stacked_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("wk_weights_proj", "wk", 0),
            ("wk_weights_proj", "weights_proj", 1),
        ]
        pending_wk_mxfp8: dict[str, dict[str, torch.Tensor]] = {}
        pending_wk_fp8: dict[str, dict[str, torch.Tensor]] = {}
        pp_missing_layer_names = get_pp_missing_layer_names(self)
        skip_topk_layers = compute_skip_topk_layers(self.config)
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        local_heads = self.config.num_attention_heads // tp_size

        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        for name, loaded_weight in weights:
            if name.startswith("model.mtp_layers."):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if is_skip_topk_indexer_weight(name, skip_topk_layers):
                continue
            if _try_load_mxfp8_indexer_wk(
                name,
                loaded_weight,
                pending_wk_mxfp8,
                params_dict,
                loaded_params,
                pp_missing_layer_names,
            ):
                continue
            if _try_load_fp8_indexer_wk(
                name,
                loaded_weight,
                pending_wk_fp8,
                params_dict,
                loaded_params,
                pp_missing_layer_names,
            ):
                continue

            if name.endswith("learnable_sink_param"):
                start = tp_rank * local_heads
                loaded_weight = loaded_weight.narrow(0, start, local_heads)

            packed_expert = self._packed_expert_target(name)
            if packed_expert is not None:
                mapped, is_gate_up = packed_expert
                if is_pp_missing_parameter(mapped, self):
                    continue
                if mapped not in params_dict:
                    raise ValueError(
                        f"HY4 packed expert tensor has no destination: {name} "
                        f"-> {mapped}"
                    )
                param = params_dict[mapped]
                if is_gate_up:
                    gate, up = loaded_weight.chunk(2, dim=-2)
                    loaded_gate = self._load_all_experts(
                        mapped,
                        param,
                        gate,
                        "w1",
                        self.config.n_routed_experts,
                    )
                    loaded_up = self._load_all_experts(
                        mapped,
                        param,
                        up,
                        "w3",
                        self.config.n_routed_experts,
                    )
                    loaded = loaded_gate or loaded_up
                else:
                    loaded = self._load_all_experts(
                        mapped,
                        param,
                        loaded_weight,
                        "w2",
                        self.config.n_routed_experts,
                    )
                if not loaded:
                    raise ValueError(f"No local HY4 expert accepted {name}")
                loaded_params.add(mapped)
                continue

            # Per-expert weight loading (checkpoint format: mlp.experts.{id}.{proj})
            if "mlp.experts." in name and packed_expert is None:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if name_mapped not in params_dict:
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                    break
                if is_expert_weight:
                    continue

            mapped_stacked = False
            for param_name, weight_name, shard_id in stacked_mapping:
                if weight_name not in name or "mlp.experts" in name:
                    continue
                if weight_name in ("wk", "weights_proj") and ".indexer." not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(mapped, self):
                    mapped_stacked = True
                    break
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                mapped_stacked = True
                break
            if mapped_stacked:
                continue

            if is_pp_missing_parameter(name, self):
                continue
            if name not in params_dict:
                raise ValueError(f"Unexpected HY4 checkpoint parameter: {name}")
            param = params_dict[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, loaded_weight)
            loaded_params.add(name)

        if pending_wk_mxfp8 or pending_wk_fp8:
            raise ValueError(
                "HY4 checkpoint coverage failed: incomplete FP8/MXFP8 indexer "
                "wk weight/scale pairs for "
                f"{sorted(set(pending_wk_mxfp8) | set(pending_wk_fp8))}"
            )

        runtime_attention_scales = (
            ".self_attn.attn.q_scale",
            ".self_attn.attn.k_scale",
            ".self_attn.attn.v_scale",
            ".self_attn.attn.prob_scale",
            ".self_attn.mla_attn.q_scale",
            ".self_attn.mla_attn.k_scale",
            ".self_attn.mla_attn.v_scale",
            ".self_attn.mla_attn.prob_scale",
        )
        missing = sorted(
            name
            for name in set(params_dict).difference(loaded_params)
            if not name.endswith(runtime_attention_scales)
        )
        if missing:
            preview = ", ".join(missing[:20])
            suffix = " ..." if len(missing) > 20 else ""
            raise ValueError(
                f"HY4 checkpoint coverage failed: {len(missing)} model "
                f"parameters were not loaded: {preview}{suffix}"
            )
        return loaded_params


__all__ = ["HYV4ForCausalLM"]
