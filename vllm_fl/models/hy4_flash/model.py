# SPDX-License-Identifier: Apache-2.0
"""HYV4 Flash: hybrid sparse/window attention, IndexPool and Engram."""

from collections.abc import Iterable

import torch

from vllm.config import VllmConfig
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.platforms import current_platform

from vllm_fl.models.hy4.model import (
    HYV4DecoderLayer,
    HYV4ForCausalLM,
    HYV4HyperLayer,
    HYV4Model,
)

from .attention import HYV4FlashAttention
from .engram import FlashEngram, FlashNgramHashState
from .moe import HYV4FlashDenseMLP, HYV4FlashMoE


def validate_runtime(vllm_config: VllmConfig) -> None:
    config = vllm_config.model_config.hf_text_config
    config.validate_flash_config()
    parallel = vllm_config.parallel_config
    if parallel.pipeline_parallel_size != 1 or parallel.data_parallel_size != 1:
        raise ValueError("HYV4 Flash currently supports only single-node TP+EP")
    if not current_platform.is_cuda_alike():
        raise NotImplementedError("HYV4 Flash requires a CUDA-alike runtime")
    if not vllm_config.use_v2_model_runner:
        raise ValueError("HYV4 Flash Engram requires the V2 model runner")
    if vllm_config.speculative_config is not None:
        raise ValueError("HYV4 Flash DFly is not enabled for this adaptation")
    if vllm_config.model_config.dtype != torch.bfloat16:
        raise ValueError("HYV4 Flash requires bfloat16 compute")
    for field in ("num_attention_heads", "swa_num_attention_heads"):
        if getattr(config, field) % parallel.tensor_parallel_size:
            raise ValueError(f"{field} must be divisible by tensor_parallel_size")


class HYV4FlashDecoderLayer(HYV4DecoderLayer):
    def __init__(self, vllm_config, prefix, topk_indices_buffer, runtime_plan):
        # Do not call the sparse-only parent constructor: its attention and MoE
        # have different layouts.  iHC and its two RMS norms are shared.
        torch.nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        layer_idx = int(prefix.rsplit(".", 1)[-1])
        self.hc_attn_layer = HYV4HyperLayer(config)
        self.self_attn = HYV4FlashAttention(
            vllm_config, config, layer_idx, f"{prefix}.self_attn",
            topk_indices_buffer, runtime_plan,
        )
        self.hc_mlp_layer = HYV4HyperLayer(config)
        self.input_layernorm = self._make_norm(config)
        self.post_attention_layernorm = self._make_norm(config)
        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = HYV4FlashMoE(config, vllm_config, f"{prefix}.mlp")
        else:
            self.mlp = HYV4FlashDenseMLP(config, vllm_config.quant_config, f"{prefix}.mlp")

    @staticmethod
    def _make_norm(config):
        from vllm.model_executor.layers.layernorm import RMSNorm
        return RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class HYV4FlashModel(HYV4Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # HYV4Model allocates the top-k buffer and validates the sparse runtime.
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = self.config
        self.start_layer, self.end_layer = 0, config.num_hidden_layers
        self.layers = torch.nn.ModuleList([
            HYV4FlashDecoderLayer(
                vllm_config, f"{prefix}.layers.{index}",
                self.topk_indices_buffer, self.runtime_plan,
            )
            for index in range(config.num_hidden_layers)
        ])
        self.engram_hash = None
        if config.engram_layer_ids:
            self.engram_hash = FlashNgramHashState(
                config, dummy=vllm_config.load_config.load_format == "dummy"
            )
            for layer_id in config.engram_layer_ids:
                self.layers[layer_id - 1].engram_layer = FlashEngram(
                    vllm_config, self.engram_hash.layout, layer_id,
                    f"{prefix}.layers.{layer_id - 1}.engram_layer",
                )
        self.token_lookback_depth = config.engram_max_ngram_size - 1 if self.engram_hash else 0

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None,
                lookback_token_ids=None):
        if self.engram_hash is None:
            return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)
        if input_ids is None:
            raise ValueError("Flash Engram requires raw input token IDs")
        hidden = self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        streams = hidden.unsqueeze(1).expand(-1, self.config.hc_mult, -1)
        hashes = input_ids.new_empty(
            (len(self.config.engram_layer_ids), input_ids.numel(), self.engram_hash.offsets.shape[1]),
            dtype=torch.int32,
        )
        self.engram_hash.prepare_hashes_with_output(
            input_ids, lookback_token_ids, self.layers[0].self_attn.mla_attn.layer_name, hashes,
        )
        hash_indices = {layer_id - 1: index for index, layer_id in enumerate(self.config.engram_layer_ids)}
        for index, layer in enumerate(self.layers):
            if index in hash_indices:
                streams = layer.engram_layer(streams, hashes[hash_indices[index]])
            streams = layer(positions, streams)
        return self.norm(self.hc_head(streams))

    def _target_weights(self, weights):
        for name, value in weights:
            if name.startswith(("model.dfly_layers.", "model.mtp_layers.")):
                continue
            if name.startswith("model.layers."):
                parts = name.split(".")
                if parts[2].isdigit() and int(parts[2]) >= self.config.num_hidden_layers:
                    continue
            yield name, value

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(self._target_weights(weights))


class HYV4FlashForCausalLM(HYV4ForCausalLM):
    supports_pp = False
    requires_raw_input_tokens = True
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_experts.gate_proj": (".shared_experts.gate_up_proj", 0),
            ".shared_experts.up_proj": (".shared_experts.gate_up_proj", 1),
        },
        orig_to_new_substr={".compressed_indexer.": ".indexer."},
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        validate_runtime(vllm_config)
        torch.nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_text_config
        self.quant_config = vllm_config.quant_config
        self.model = HYV4FlashModel(vllm_config=vllm_config, prefix=f"{prefix}.model")
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        self.lm_head = ParallelLMHead(
            self.config.vocab_size, self.config.hidden_size,
            quant_config=self.quant_config, prefix=f"{prefix}.lm_head",
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.token_lookback_depth = self.model.token_lookback_depth
        self.moe_mlp_layers = [
            layer.mlp for layer in self.model.layers
            if isinstance(layer.mlp, HYV4FlashMoE)
        ]
        self.moe_layers = [layer.experts.routed_experts for layer in self.moe_mlp_layers]
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        example_moe = self.moe_mlp_layers[0] if self.moe_mlp_layers else None
        if example_moe is not None:
            self.extract_moe_parameters(example_moe)

    @staticmethod
    def get_model_state_cls():
        from vllm_fl.worker.flash_model_state import FlashModelState
        return FlashModelState

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None,
                lookback_token_ids=None):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds, lookback_token_ids)

    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)
