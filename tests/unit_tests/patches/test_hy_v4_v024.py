from types import SimpleNamespace

import torch

from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from vllm_fl.model_loader.hy_v4_loader import (
    HYV4SafetensorsLoader,
    _contiguous_runs,
    _read_local_experts,
)
from vllm_fl.models.hy_v4 import HYV4ForCausalLM, _try_load_mxfp8_indexer_wk
from vllm_fl.patches import hy_v4_v024 as compat


def test_hy4_convertor_uses_compressed_mla_dimensions():
    config = SimpleNamespace(
        model_type="hy_v4",
        architectures=["HYV4ForCausalLM"],
        hidden_size=6144,
        num_hidden_layers=78,
        num_attention_heads=64,
        num_key_value_heads=8,
        vocab_size=120832,
        n_routed_experts=256,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        max_position_embeddings=1048576,
        quantization_config=None,
    )
    converted = compat.HYV4ModelArchConfigConvertor(config, config).convert()

    assert converted.head_size == 576
    assert converted.total_num_kv_heads == 8
    assert converted.is_deepseek_mla is True


def test_hy4_convertor_normalizes_mxfp8_for_override_detection():
    quant_config = {
        "quant_method": "mxfp8",
        "ignored_layers": ["lm_head"],
        "kv_cache_quant_algo": None,
    }
    config = SimpleNamespace(
        model_type="hy_v4",
        quantization_config=quant_config,
    )
    converted = compat.HYV4ModelArchConfigConvertor(
        config, config
    ).get_quantization_config()

    assert converted == {
        "quant_method": "modelopt",
        "quantization": {
            "quant_algo": "MXFP8",
            "kv_cache_quant_algo": None,
            "exclude_modules": ["lm_head"],
        },
    }
    assert quant_config["quant_method"] == "mxfp8"


def test_mxfp8_alias_is_probed_after_canonical_override():
    class FakeMXFP8Config:
        @classmethod
        def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
            return "modelopt_mxfp8"

    class OtherConfig:
        pass

    configs = {
        "mxfp8": FakeMXFP8Config,
        "modelopt_mxfp8": FakeMXFP8Config,
        "other": OtherConfig,
    }
    quantization = SimpleNamespace(
        get_quantization_config=lambda name: configs[name]
    )

    compat._patch_mxfp8_override_order(quantization)
    alias = quantization.get_quantization_config("mxfp8")
    canonical = quantization.get_quantization_config("modelopt_mxfp8")

    assert alias.override_quantization_method({}, None) is None
    assert canonical.override_quantization_method({}, None) == "modelopt_mxfp8"
    assert quantization.get_quantization_config("other") is OtherConfig

    # Re-applying the runtime hook must not stack wrappers.
    getter = quantization.get_quantization_config
    compat._patch_mxfp8_override_order(quantization)
    assert quantization.get_quantization_config is getter


def test_apply_registers_plugin_owned_hy4_components(monkeypatch):
    from vllm.model_executor import model_loader
    from vllm.model_executor.models import registry as model_registry
    from vllm.transformers_utils import (
        config as transformers_config,
        model_arch_config_convertor,
    )

    registered_models = {}
    fake_registry = SimpleNamespace(
        register_model=lambda architecture, model: registered_models.__setitem__(
            architecture, model
        )
    )
    registered_loaders = {}

    def register_loader(load_format):
        def register(loader):
            registered_loaders[load_format] = loader
            model_loader._LOAD_FORMAT_TO_MODEL_LOADER[load_format] = loader
            return loader

        return register

    monkeypatch.setattr(compat, "is_vllm_024", lambda: True)
    monkeypatch.setattr(transformers_config, "_CONFIG_REGISTRY", {})
    monkeypatch.setattr(
        model_arch_config_convertor, "MODEL_ARCH_CONFIG_CONVERTORS", {}
    )
    monkeypatch.setattr(model_registry, "ModelRegistry", fake_registry)
    monkeypatch.setattr(model_loader, "_LOAD_FORMAT_TO_MODEL_LOADER", {})
    monkeypatch.setattr(model_loader, "register_model_loader", register_loader)

    assert compat.apply_hy_v4_v024_patches() is True

    assert {"hy_v4": compat.HYV4Config} == transformers_config._CONFIG_REGISTRY
    assert {
        "hy_v4": compat.HYV4ModelArchConfigConvertor
    } == model_arch_config_convertor.MODEL_ARCH_CONFIG_CONVERTORS
    assert registered_models == {
        "HYV4ForCausalLM": "vllm_fl.models.hy_v4:HYV4ForCausalLM"
    }
    assert registered_loaders == {
        "hy4_safetensors": compat.HYV4SafetensorsLoader
    }


def test_flagos_oot_platform_inherits_mxfp8_linear_candidates():
    from vllm.model_executor.kernels.linear import _POSSIBLE_MXFP8_KERNELS
    from vllm.model_executor.kernels.linear.mxfp8.emulation import (
        EmulationMxfp8LinearKernel,
    )
    from vllm.platforms import PlatformEnum

    from vllm_fl.quantization.quant_linear import add_oot_quant_kernel

    add_oot_quant_kernel()

    assert PlatformEnum.OOT in _POSSIBLE_MXFP8_KERNELS
    assert EmulationMxfp8LinearKernel in _POSSIBLE_MXFP8_KERNELS[PlatformEnum.OOT]


def test_plugin_loader_delegates_discovery_as_safetensors(monkeypatch):
    loader = object.__new__(HYV4SafetensorsLoader)
    loader.load_config = SimpleNamespace(load_format="hy4_safetensors")
    observed = []

    def fake_prepare(self, *args):
        observed.append(self.load_config.load_format)
        return "/weights", ["/weights/model.safetensors"], True

    monkeypatch.setattr(DefaultModelLoader, "_prepare_weights", fake_prepare)
    result = loader._prepare_weights("/weights", None, None, False, None)

    assert result == ("/weights", ["/weights/model.safetensors"], True)
    assert observed == ["safetensors"]
    assert loader.load_config.load_format == "hy4_safetensors"


def test_hy4_contiguous_expert_runs_and_slices():
    assert _contiguous_runs([]) == []
    assert _contiguous_runs([0, 1, 2, 7, 9, 10]) == [
        (0, 3),
        (7, 8),
        (9, 11),
    ]
    packed = torch.arange(12).view(6, 2)
    torch.testing.assert_close(
        _read_local_experts(packed, [1, 2, 5]),
        packed[[1, 2, 5]],
    )


def test_hy4_packed_experts_do_not_match_shared_expert():
    shared = "model.layers.1.mlp.shared_experts.down_proj.weight"
    routed = "model.layers.1.mlp.experts.down_proj"
    assert HYV4ForCausalLM._packed_expert_target(shared) is None
    assert HYV4ForCausalLM._packed_expert_target(routed) == (
        "model.layers.1.mlp.experts.routed_experts.w2_weight",
        False,
    )


def test_hy4_mxfp8_indexer_wk_is_dequantized_into_fused_projection():
    param = torch.nn.Parameter(torch.zeros(4, 64, dtype=torch.bfloat16))

    def load_shard(target, weight, shard_id):
        assert shard_id == 0
        with torch.no_grad():
            target[: weight.shape[0]].copy_(weight)

    param.weight_loader = load_shard
    params = {"model.layers.0.self_attn.indexer.wk_weights_proj.weight": param}
    pending = {}
    loaded = set()
    prefix = "model.layers.0.self_attn.indexer.wk"
    scale = torch.tensor([[127, 128], [126, 127]], dtype=torch.uint8)
    weight = torch.ones(2, 64, dtype=torch.float8_e4m3fn)

    assert _try_load_mxfp8_indexer_wk(
        f"{prefix}.weight_scale", scale, pending, params, loaded
    )
    assert _try_load_mxfp8_indexer_wk(
        f"{prefix}.weight", weight, pending, params, loaded
    )

    expected = torch.tensor(
        [[1.0] * 32 + [2.0] * 32, [0.5] * 32 + [1.0] * 32],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(param[:2], expected)
    assert not pending
    assert loaded == set(params)


def test_hy4_mxfp8_indexer_wk_skips_nonlocal_pipeline_stage():
    pending = {}
    loaded = set()
    prefix = "model.layers.41.self_attn.indexer.wk"
    missing = ["model.layers.41."]

    assert _try_load_mxfp8_indexer_wk(
        f"{prefix}.weight_scale",
        torch.ones(2, 2, dtype=torch.uint8),
        pending,
        {},
        loaded,
        missing,
    )
    assert _try_load_mxfp8_indexer_wk(
        f"{prefix}.weight",
        torch.ones(2, 64, dtype=torch.float8_e4m3fn),
        pending,
        {},
        loaded,
        missing,
    )
    assert not pending
    assert not loaded
