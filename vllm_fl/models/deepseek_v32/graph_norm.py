# SPDX-License-Identifier: Apache-2.0
"""Torch.library tracing boundaries; numerical work remains public FlagGems."""

import torch
from flag_gems.modules.normalization import gems_rms_forward


@torch.library.custom_op("vllm_fl::ppu_graph_rms_norm", mutates_args=())
def ppu_graph_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    return gems_rms_forward(x, None, weight, eps)


@ppu_graph_rms_norm.register_fake
def _fake_rms(x, weight, eps):
    # Current public FlagGems RMS makes x contiguous then empty_like(x).
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


@torch.library.custom_op(
    "vllm_fl::ppu_graph_skip_rms_norm", mutates_args=("x", "residual")
)
def ppu_graph_skip_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    result = gems_rms_forward(x, residual, weight, eps)
    if result[0] is not x or result[1] is not residual:
        raise RuntimeError(
            "Existing skip RMS must return its already-contiguous inputs"
        )


@ppu_graph_skip_rms_norm.register_fake
def _fake_skip(x, residual, weight, eps):
    return None


def graph_rms_forward(x, residual, weight, eps):
    if residual is None:
        return ppu_graph_rms_norm(x, weight, eps)
    # Same copies as the existing public implementation, before explicit mutations.
    x = x.contiguous()
    residual = residual.contiguous()
    ppu_graph_skip_rms_norm(x, residual, weight, eps)
    return x, residual
