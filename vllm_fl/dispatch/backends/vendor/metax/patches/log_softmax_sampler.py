# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# FlagGems' MetaX log_softmax Triton kernel uses BLOCK_N = next_power_of_2(N).
# For large vocabularies (e.g. 152064 -> BLOCK_N=262144), the pointer arithmetic
# overflows 32-bit addressing on the MACA backend:
#   "RuntimeError: Triton Error [MACA]: memory size or pointer value too large
#    to fit in 32 bit"
# Patch compute_logprobs to use a manual log_softmax that decomposes into
# element-wise ops which don't trigger the problematic kernel.

import torch
import vllm.v1.sample.sampler as sampler_module


def _compute_logprobs_no_triton(logits: torch.Tensor) -> torch.Tensor:
    logits_f32 = logits.to(torch.float32)
    max_logits = logits_f32.max(dim=-1, keepdim=True).values
    shifted = logits_f32 - max_logits
    log_sum_exp = shifted.exp().sum(dim=-1, keepdim=True).log()
    return shifted - log_sum_exp


sampler_module.Sampler.compute_logprobs = staticmethod(_compute_logprobs_no_triton)
