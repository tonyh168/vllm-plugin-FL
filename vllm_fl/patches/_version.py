"""vLLM version gate for plugin patches.

The GLM5-Next runtime patches were authored against vLLM 0.24 semantics
(hybrid-cache config, model_arch_config_convertor, and the 0.24 KV plumbing).
``is_vllm_024`` guards those registrations so they only install on a matching
runtime.  On this deployment the vLLM package reports a dev/empty build string
(e.g. ``0.1.dev...+gee0da84ab.empty``) whose git description resolves to the
v0.24.0 tag, so the gate treats a 0.24 series — or an indeterminate dev build
that still exposes the 0.24 APIs — as 0.24.
"""

import logging

logger = logging.getLogger(__name__)


def _vllm_version_string() -> str:
    try:
        from importlib.metadata import version

        return version("vllm")
    except Exception:
        pass
    try:
        import vllm
    except Exception:
        return ""
    return getattr(vllm, "__version__", "")


def is_vllm_024() -> bool:
    """Return True when the active vLLM exposes the 0.24 patch surface.

    A clean ``0.24.x`` release string matches directly.  Editable/empty dev
    builds (``0.1.devN+g<sha>.empty``) do not carry a usable release number, so
    fall back to probing for the 0.24-era APIs the GLM5-Next patches require.
    """
    ver = _vllm_version_string()
    if ver.startswith("0.24.") or ver == "0.24.0":
        return True
    try:
        from vllm.model_executor.models.config import (
            HybridAttentionMambaModelConfig,  # noqa: F401
        )
        from vllm.transformers_utils.model_arch_config_convertor import (
            MODEL_ARCH_CONFIG_CONVERTORS,  # noqa: F401
            ModelArchConfigConvertorBase,  # noqa: F401
        )

        return True
    except Exception as exc:
        logger.debug("is_vllm_024 probe failed: %s", exc)
        return False
