# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 @ 真武/平头哥 PPU 的本平台补丁(叠在 plugin fork 030plugin-for-day0 之上)。"""

from .fp8_moe_oracle import install_fl_fp8_moe_selector

__all__ = ["install_fl_fp8_moe_selector", "TritonFp8ExpertsFL"]


def __getattr__(name):
    if name == "TritonFp8ExpertsFL":
        from .fp8_moe_experts import TritonFp8ExpertsFL

        return TritonFp8ExpertsFL
    raise AttributeError(name)
