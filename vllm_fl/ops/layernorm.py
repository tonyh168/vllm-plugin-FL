# Copyright (c) 2025 BAAI. All rights reserved.

from typing import Optional, Union
import torch
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm_fl.dispatch import CachedOp

_rms_norm = CachedOp("rms_norm")
_ppu_graph_rms = None


def enable_ppu_graph_rms_norm() -> None:
    """Enable only from the validated DeepSeek PPU PIECEWISE activation plan."""
    global _ppu_graph_rms
    from vllm_fl.models.deepseek_v32.graph_norm import graph_rms_forward

    _ppu_graph_rms = graph_rms_forward


class RMSNormFL(RMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if _ppu_graph_rms is not None:
            return _ppu_graph_rms(x, residual, self.weight, self.variance_epsilon)
        return _rms_norm(self, x, residual)


__all__ = ["RMSNormFL"]
