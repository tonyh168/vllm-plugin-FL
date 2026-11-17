# SPDX-License-Identifier: Apache-2.0
"""Flash dense/latent MoE layers on the existing MetaX W8A8 path."""

import torch

from vllm.model_executor.layers.fused_moe import fused_moe as fused_moe_layers
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_fl.models.hy4.model import HYV4DenseMLP, HYV4MoE


class HYV4FlashDenseMLP(HYV4DenseMLP):
    def __init__(self, config, quant_config, prefix):
        super().__init__(config.hidden_size, config.intermediate_size,
                         config.hidden_act, quant_config, prefix=prefix)
        self.post_fc2_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x):
        return self.post_fc2_norm(super().forward(x))


class HYV4FlashMoE(HYV4MoE):
    def __init__(self, config, vllm_config, prefix):
        latent = config.routed_expert_hidden_size
        self.routed_expert_down_proj = ReplicatedLinear(
            config.hidden_size, latent, bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.routed_expert_down_proj",
        )
        self.routed_expert_up_proj = ReplicatedLinear(
            latent, config.hidden_size, bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.routed_expert_up_proj",
        )
        super().__init__(config, vllm_config, prefix)
        parallel_config = vllm_config.parallel_config
        # FusedMoE owns both latent projections: the router and shared expert
        # continue to see the full-width residual stream.
        self.experts = fused_moe_layers.FusedMoE(
            gate=self.gate, num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok, hidden_size=latent,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob, quant_config=vllm_config.quant_config,
            use_grouped_topk=True, num_expert_group=config.n_group,
            topk_group=config.topk_group, prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            routed_scaling_factor=config.routed_scaling_factor,
            swiglu_limit=config.swiglu_limit,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            apply_routed_scale_to_output=False,
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            router_logits_dtype=torch.float32,
            is_sequence_parallel=self.is_sequence_parallel,
            shared_experts=self.shared_experts,
            routed_input_transform=self.routed_expert_down_proj,
            routed_output_transform=self.routed_expert_up_proj,
        )
        self.n_local_physical_experts = self.experts.routed_experts.local_num_experts
        self.combine_output_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states):
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, shape[-1])
        if self.experts.is_internal_router:
            combined = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
        else:
            router_logits, _ = self.gate(hidden_states)
            combined = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        return self.combine_output_norm(combined.view(shape))
