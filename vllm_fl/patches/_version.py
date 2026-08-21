# Copyright (c) 2025 BAAI. All rights reserved.
"""vLLM version detection helpers for conditional patches."""


def is_vllm_024() -> bool:
    """Return True when running on vLLM 0.24.x or compatible empty/dev builds."""
    try:
        import vllm
        version = getattr(vllm, "__version__", "0.0.0")
        # Empty/dev builds (e.g. "0.1.dev17937+gd487ba0ab.empty") are built
        # from vLLM 0.24+ source and should use 0.24 patches.
        if "empty" in version or "dev" in version:
            return True
        major_minor = version.split(".")[:2]
        return major_minor[0] == "0" and major_minor[1].startswith("24")
    except Exception:
        return False
