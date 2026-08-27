# Copyright (c) 2025 BAAI. All rights reserved.

import importlib
import os
import logging
import sys

# torch.float4_e2m1fn_x2 exists only in CUDA builds of PyTorch 2.7+.
# vllm.ir.tolerances references it at module level, so we inject a sentinel
# before any vllm.ir import can happen.
if "torch" in sys.modules:
    _torch = sys.modules["torch"]
    if not hasattr(_torch, "float4_e2m1fn_x2"):
        _torch.float4_e2m1fn_x2 = _torch.uint8
else:
    import torch as _torch
    if not hasattr(_torch, "float4_e2m1fn_x2"):
        _torch.float4_e2m1fn_x2 = _torch.uint8
del _torch

try:
    from vllm_fl.utils import get_op_config as _get_op_config
except ModuleNotFoundError as exc:
    # The native GLM5-Next baseline reuses only this package's config/model
    # modules and deliberately does not install or activate FlagGems.
    if exc.name != "flag_gems":
        raise

    def _get_op_config():
        return None

from . import version as version  # PyTorch-style: vllm_fl.version.git_version


logger = logging.getLogger(__name__)

# Guard so the version banner prints once per process (register() is invoked
# repeatedly along the plugin-load chain).
_VERSION_LOGGED = False


def __getattr__(name):
    if name == "distributed":
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _patch_transformers_compat():
    """Patch transformers compatibility for ALLOWED_LAYER_TYPES and tokenizer."""
    import transformers.configuration_utils as cfg
    if not hasattr(cfg, "ALLOWED_LAYER_TYPES"):
        cfg.ALLOWED_LAYER_TYPES = getattr(
            cfg, "ALLOWED_ATTENTION_LAYER_TYPES", ()
        )
    # transformers>=5.9 tightened `layer_types` validation against a fixed
    # ALLOWED_LAYER_TYPES tuple that does not know GLM5-Next's DSA layer tag.
    # PretrainedConfig.validate_layer_type reads this module global, so extend
    # it in place instead of patching stock transformers (mirrors the NVIDIA
    # transformers patch, which we do not ship on OOT accelerators).
    for _lt in ("deepseek_sparse_attention",):
        if _lt not in cfg.ALLOWED_LAYER_TYPES:
            cfg.ALLOWED_LAYER_TYPES = tuple(cfg.ALLOWED_LAYER_TYPES) + (_lt,)


def _register_flagcx_connector():
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    for _alias in ("FlagCXConnector", "FlagcxConnector"):
        if _alias not in KVConnectorFactory._registry:
            KVConnectorFactory.register_connector(
                _alias,
                "vllm_fl.distributed.kv_transfer.flagcx_connector",
                "FlagCXConnector",
            )


def _patch_flash_attn_import():
    """Stub vllm.vllm_flash_attn if CUDA flash attention C extensions are missing."""
    import sys
    if "vllm.vllm_flash_attn" in sys.modules:
        return
    try:
        import vllm.vllm_flash_attn  # noqa: F401
    except ImportError:
        import types
        stub = types.ModuleType("vllm.vllm_flash_attn")
        stub.FA2_AVAILABLE = False
        stub.FA3_AVAILABLE = False
        stub.fa_version_unsupported_reason = lambda *a, **kw: "flash_attn C extensions not available"
        stub.flash_attn_varlen_func = None
        stub.get_scheduler_metadata = None
        stub.is_fa_version_supported = lambda *a, **kw: False
        sys.modules["vllm.vllm_flash_attn"] = stub


def _patch_custom_ops():
    """Register fallback schemas when neither vLLM extension ABI is present."""
    for module_name in ("vllm._C", "vllm._C_stable_libtorch"):
        try:
            importlib.import_module(module_name)
            return
        except (ImportError, OSError):
            continue

    try:
        import vllm_fl._C  # noqa: F401
    except (ImportError, OSError) as e:
        logger.debug("Failed to import vllm_fl._C: %s", e)

    from vllm_fl.ops._C_ops_registry import register_op_schemas
    register_op_schemas()


def register():
    """Register the FL platform."""
    # Print the plugin-FL code version at the very first entry point every
    # process (incl. each spawned Ray worker) hits when loading this plugin.
    # Worker-class hooks proved unreliable for this, so anchor it here where
    # execution is guaranteed. Emitted to stdout with flush so it survives
    # regardless of logger level / Ray log capture.
    global _VERSION_LOGGED
    if not _VERSION_LOGGED:
        _VERSION_LOGGED = True
        try:
            from vllm_fl.utils import log_plugin_fl_version
            log_plugin_fl_version()
        except Exception as _e:
            print(f"[plugin-FL version] <failed to compute: {_e}>", flush=True)

    _patch_custom_ops()
    _patch_flash_attn_import()
    _patch_transformers_compat()

    # Model-specific platform patches
    from vllm_fl.patches.glm_moe_dsa import apply_platform_patches as glm5_platform
    glm5_platform()

    # Note: FlagCX connector registration is deferred to register_model()
    # to avoid circular imports during VllmConfig.__post_init__ in spawned
    # subprocesses.

    multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if multiproc_method is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    _get_op_config()

    return "vllm_fl.platform.PlatformFL"

def register_quant_linear():
    from vllm.platforms import current_platform
    # vllm.model_executor.kernels.linear triggers cutlass_scaled_mm_supports_fp8
    # at module level, which requires torch.ops._C — not available on MUSA.
    if current_platform.device_type == "musa":
        return
    from vllm_fl.quantization.quant_linear import add_oot_quant_kernel
    add_oot_quant_kernel()

def register_router():
    from vllm.platforms import current_platform
    # fused_moe import chain triggers cutlass_scaled_mm_supports_fp8 on MUSA
    if current_platform.device_type == "musa":
        return
    from vllm_fl.utils import is_oot_enabled
    if not is_oot_enabled():
        return
    from vllm_fl.ops.fused_moe.router import replace_router_with_fl
    replace_router_with_fl()

def register_model():
    """Register FL-specific models not yet upstream."""
    # General plugins are loaded independently in spawned model-inspection and
    # worker processes, so all runtime compatibility hooks must be idempotent.
    from vllm_fl.patches.moe_sum import patch_vllm_moe_sum
    from vllm_fl.patches.qwen3_5_text import apply_qwen3_5_text_patches

    apply_qwen3_5_text_patches()
    patch_vllm_moe_sum()

    # Register the plugin-owned GLM5-Next runtime (config, arch convertor,
    # kpool/DSA KV plumbing, mHC OOT dispatch). Self-gated on vLLM 0.24.
    from vllm_fl.patches._version import is_vllm_024
    if is_vllm_024():
        from vllm_fl.patches.glm5_next_v024 import (
            apply_glm5_next_v024_patches,
        )
        apply_glm5_next_v024_patches()

    _register_flagcx_connector()

    # Register OOT quant kernels so kernel selection can find them
    register_quant_linear()
    register_router()

    # Register GLM-5 (GlmMoeDsa) — config not yet upstream
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_fl.configs.glm_moe_dsa import GlmMoeDsaConfig
        _CONFIG_REGISTRY["glm_moe_dsa"] = GlmMoeDsaConfig

        #from vllm_fl.patches.glm_moe_dsa import apply_model_patches as glm5_model
        #glm5_model()
    except Exception as e:
        logger.error(f"Register GlmMoeDsa model error: {str(e)}")
