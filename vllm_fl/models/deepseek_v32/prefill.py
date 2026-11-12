# SPDX-License-Identifier: Apache-2.0
"""PPU capability adapter for the public portable varlen attention operator."""

from vllm.platforms import current_platform
from vllm_fl.attention.mla_prefill import FlagGemsMLAPrefillBackend
from vllm_fl.dispatch.capabilities import get_flaggems_vllm_operator
from vllm_fl.utils import use_flaggems_op


class PpuMLAPrefillBackend(FlagGemsMLAPrefillBackend):
    @staticmethod
    def get_name():
        return "FLAGGEMS_PPU_MLA_PREFILL"

    @classmethod
    def is_available(cls):
        return (
            getattr(current_platform, "vendor_name", None) == "thead"
            and use_flaggems_op("flash_attn_varlen_func")
            and callable(get_flaggems_vllm_operator("flash_attn_varlen_func"))
        )

    @staticmethod
    def _operator_factory():
        if not PpuMLAPrefillBackend.is_available():
            raise RuntimeError("PPU MLA prefill requires the public FlagGems-vllm varlen operator")
        return get_flaggems_vllm_operator("flash_attn_varlen_func")
