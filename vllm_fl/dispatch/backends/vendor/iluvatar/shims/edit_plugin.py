import pathlib, sys

ROOT = pathlib.Path("/workspace/vllm-plugin-FL")
errors = []

def edit(rel, old, new, count=1):
    p = ROOT / rel
    s = p.read_text()
    if old not in s:
        errors.append(f"pattern not found in {rel}")
        return
    p.write_text(s.replace(old, new, count))
    print("edited", rel)

# 1) _patch_custom_ops: run FlagGems patch_empty_vllm before schema registration
edit("vllm_fl/__init__.py",
"""    try:
        import vllm_fl._C  # noqa: F401
    except (ImportError, OSError) as e:
        logger.debug("Failed to import vllm_fl._C: %s", e)

    from vllm_fl.ops._C_ops_registry import register_op_schemas
""",
"""    try:
        import vllm_fl._C  # noqa: F401
    except (ImportError, OSError) as e:
        logger.debug("Failed to import vllm_fl._C: %s", e)

    # Iluvatar empty wheel: bind FlagGems Triton implementations (silu,
    # rms_norm, ...) BEFORE schema registration.  The registry installs a
    # CUDA fallback for apply_repetition_penalties_ that would otherwise
    # collide inside flag_gems.patch_empty_vllm().
    try:
        from vllm_fl.utils import DeviceInfo

        if DeviceInfo().vendor_name == "iluvatar":
            import flag_gems

            flag_gems.patch_empty_vllm()
    except Exception as exc:  # pragma: no cover - optional acceleration
        logger.warning("[vllm_fl] FlagGems patch_empty_vllm failed: %s", exc)

    from vllm_fl.ops._C_ops_registry import register_op_schemas
""")

# 2) register Hy4-Flash for iluvatar + install iluvatar shims
edit("vllm_fl/__init__.py",
"""    if vendor_name == "nvidia":
        from vllm_fl.patches.minicpmo_audio import patch_minicpmo_audio_budget

        # General plugins run before encoder budgets are built in the engine
        # frontend and spawned workers, for both upstream model runners.
        patch_minicpmo_audio_budget()
        from vllm_fl.models.hy_v4_flash.registration import register_hy4_flash

        register_hy4_flash()
""",
"""    if vendor_name == "nvidia":
        from vllm_fl.patches.minicpmo_audio import patch_minicpmo_audio_budget

        # General plugins run before encoder budgets are built in the engine
        # frontend and spawned workers, for both upstream model runners.
        patch_minicpmo_audio_budget()

    if vendor_name in ("nvidia", "iluvatar"):
        from vllm_fl.models.hy_v4_flash.registration import register_hy4_flash

        register_hy4_flash()

    if vendor_name == "iluvatar":
        # Empty wheel has no vllm._C FlashMLA/DeepGEMM; importing the shims
        # binds FlagGems-vllm Triton kernels in their place.
        from vllm_fl.dispatch.backends.vendor.iluvatar.shims import (  # noqa: F401
            _fl_iluvatar_deepgemm_ops,
            _fl_iluvatar_empty_ops,
            _fl_iluvatar_gems_flashmla,
        )
""")

# 3) transformers 5.x flat rope defaults in per-layer configs
edit("vllm_fl/models/hy_v4_flash/configs.py",
"""        if nested:
            rope.pop("mrope_section", None)
            rope.pop("mrope_interleaved", None)
""",
"""        if nested:
            rope.pop("mrope_section", None)
            rope.pop("mrope_interleaved", None)
            # transformers >= 5.x 会在 __post_init__ 里把默认平铺 rope 参数
            # （如 rope_theta=10000.0）塞回 rope_parameters；per-layer 配置下应丢弃。
            for flat_key in ("rope_theta", "rope_type", "partial_rotary_factor"):
                rope.pop(flat_key, None)
""")

# 4) attention capability gate
edit("vllm_fl/models/hy_v4_flash/nvidia/attention.py",
"""        capability = current_platform.get_device_capability()
        if capability is None or capability.major not in (9, 10):
""",
"""        capability = current_platform.get_device_capability()
        from vllm_fl.utils import DeviceInfo

        if DeviceInfo().vendor_name not in ("metax", "iluvatar") and (
            capability is None or capability.major not in (9, 10)
        ):
""")

# 5) compressed indexer SM90 / DeepGEMM gates
edit("vllm_fl/models/hy_v4_flash/nvidia/compressed_indexer.py",
"""        if not current_platform.is_device_capability_family(90):
            raise NotImplementedError(
                "This Flash plugin backport supports H100 SM90 FP8 scales only"
            )
        if not has_deep_gemm():
            raise RuntimeError("Flash IndexPool requires DeepGEMM paged MQA")
""",
"""        from vllm_fl.utils import DeviceInfo

        if DeviceInfo().vendor_name not in ("metax", "iluvatar"):
            if not current_platform.is_device_capability_family(90):
                raise NotImplementedError(
                    "This Flash plugin backport supports H100 SM90 FP8 scales only"
                )
            if not has_deep_gemm():
                raise RuntimeError("Flash IndexPool requires DeepGEMM paged MQA")
""")

# 6) runtime validation: non-CUDA platforms
edit("vllm_fl/models/hy_v4_flash/nvidia/model.py",
"""    if not current_platform.is_cuda():
        raise NotImplementedError("HYV4 Flash requires NVIDIA CUDA")
""",
"""    from vllm_fl.utils import DeviceInfo

    if not current_platform.is_cuda() and DeviceInfo().vendor_name not in (
        "metax",
        "iluvatar",
    ):
        raise NotImplementedError("HYV4 Flash requires NVIDIA CUDA")
""")

# 7) skip the removed 0.32 dfly FP8-block validation on non-NVIDIA
edit("vllm_fl/models/hy_v4_flash/nvidia/model.py",
"""    if quantization == "compressed-tensors":
        from vllm.models.hy_v4.dfly_config import _validate_block_fp8_config

        _validate_block_fp8_config(
            vllm_config.model_config.model_arch_config.quantization_config
        )
""",
"""    if quantization == "compressed-tensors":
        from vllm_fl.utils import DeviceInfo

        if DeviceInfo().vendor_name not in ("metax", "iluvatar"):
            # vLLM 0.32 移除了 vllm.models.hy_v4.dfly_config；该权重是 INT8
            # W8A8 (compressed-tensors int-quantized)，FP8 block 校验不适用。
            from vllm.models.hy_v4.dfly_config import _validate_block_fp8_config

            _validate_block_fp8_config(
                vllm_config.model_config.model_arch_config.quantization_config
            )
""")

if errors:
    print("ERRORS:", errors)
    sys.exit(1)
print("ALL_EDITS_OK")
