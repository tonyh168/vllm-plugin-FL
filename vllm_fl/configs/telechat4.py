# SPDX-License-Identifier: Apache-2.0
"""TeleChat4 config bridge for vLLM plugin.

transformers does not recognise model_type ``telechat4``.
This config bridge lets vLLM load the HuggingFace checkpoint by
inheriting from DeepseekV2Config and setting the correct model_type.
"""

from transformers import DeepseekV2Config


class TeleChat4Config(DeepseekV2Config):
    model_type = "telechat4"

    def __init__(
        self,
        num_residual_streams=1,
        mhc_sinkhorn_iterations=20,
        mhc_init_gating_factor=0.01,
        mhc_h_res_clamp_min=-30,
        mhc_h_res_clamp_max=30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_residual_streams = num_residual_streams
        self.mhc_sinkhorn_iterations = mhc_sinkhorn_iterations
        self.mhc_init_gating_factor = mhc_init_gating_factor
        self.mhc_h_res_clamp_min = mhc_h_res_clamp_min
        self.mhc_h_res_clamp_max = mhc_h_res_clamp_max
