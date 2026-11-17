# SPDX-License-Identifier: Apache-2.0
"""Minimal model-state interface for Flash V2 lookback input."""

from typing import Any

import torch

from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.triton_utils import tl, triton


@triton.jit
def _gather_lookback_kernel(lookback_ptr, idx_mapping_ptr, query_start_loc_ptr,
                            positions_ptr, all_token_ids_ptr, all_token_ids_stride,
                            num_reqs, DEPTH: tl.constexpr, BLOCK_DEPTH: tl.constexpr):
    batch_idx = tl.program_id(0)
    in_batch = batch_idx < num_reqs
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx, mask=in_batch, other=0)
    begin = tl.load(query_start_loc_ptr + batch_idx, mask=in_batch, other=0)
    end = tl.load(query_start_loc_ptr + batch_idx + 1, mask=in_batch, other=0)
    in_batch = in_batch & (end > begin)
    num_computed = tl.load(positions_ptr + begin, mask=in_batch, other=0)
    offs = tl.arange(0, BLOCK_DEPTH)
    pos = num_computed - 1 - offs
    valid = in_batch & (offs < DEPTH) & (pos >= 0)
    ids = tl.load(all_token_ids_ptr + req_state_idx.to(tl.int64) * all_token_ids_stride + pos,
                  mask=valid, other=-1)
    tl.store(lookback_ptr + batch_idx * DEPTH + offs, ids, mask=offs < DEPTH)


class FlashModelState(DefaultModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        depth = model.token_lookback_depth
        self.lookback_token_ids = None
        if depth:
            self.lookback_token_ids = torch.full(
                (self.max_num_reqs, depth), -1, dtype=torch.int32, device=device,
            )

    def prepare_inputs(self, input_batch: InputBatch, req_states: RequestState) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        window = self.lookback_token_ids
        if window is None:
            return model_inputs
        depth = window.shape[1]
        _gather_lookback_kernel[(window.shape[0],)](
            window, input_batch.idx_mapping, input_batch.query_start_loc,
            input_batch.positions, req_states.all_token_ids.gpu,
            req_states.all_token_ids.stride(0), input_batch.idx_mapping.shape[0],
            DEPTH=depth, BLOCK_DEPTH=triton.next_power_of_2(depth),
        )
        model_inputs["lookback_token_ids"] = window
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if self.lookback_token_ids is not None:
            self.lookback_token_ids.fill_(-1)
            model_inputs["lookback_token_ids"] = self.lookback_token_ids
        return model_inputs
