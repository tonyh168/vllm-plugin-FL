# SPDX-License-Identifier: Apache-2.0
"""Process-start provider override for GLM5-Next A/B validation."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal, cast

from vllm.platforms import current_platform

Provider = Literal["auto", "nvidia", "flaggems"]
ENV_NAME = "VLLM_FL_GLM5_PROVIDER"


@lru_cache(maxsize=1)
def get_glm5_provider() -> Provider:
    value = os.environ.get(ENV_NAME, "auto").strip().lower()
    if value not in ("auto", "nvidia", "flaggems"):
        raise ValueError(
            f"{ENV_NAME} must be one of auto|nvidia|flaggems, got {value!r}"
        )
    if value == "nvidia" and not current_platform.is_cuda():
        raise RuntimeError(
            f"{ENV_NAME}=nvidia requires an NVIDIA CUDA platform"
        )
    return cast(Provider, value)


def use_nvidia_reference() -> bool:
    provider = get_glm5_provider()
    return provider == "nvidia" or (
        provider == "auto" and current_platform.is_cuda()
    )


__all__ = ["ENV_NAME", "get_glm5_provider", "use_nvidia_reference"]
