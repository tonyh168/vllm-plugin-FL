"""Define the _C cache ops missing from vLLM's empty wheel on Iluvatar.

``flag_gems.patch_empty_vllm()`` (invoked from ``vllm_fl._patch_custom_ops``)
binds the activation/norm kernels.  The MLA query-pack op used by sparse MLA
is not covered there, so define it here with the vLLM contract.

Kill switch: ``FL_ILUVATAR_EMPTY_VLLM_OPS=0``.
"""

import os

import torch

from ._fl_iluvatar_gate import iluvatar_active

_SWITCH = "FL_ILUVATAR_EMPTY_VLLM_OPS"
_LIB = None


def _concat_mla_q(ql_nope, q_pe, q_out) -> None:
    """vLLM contract: write into the first ``ql_nope.size(1)`` heads of q_out."""
    heads = ql_nope.shape[1]
    nope = ql_nope.shape[-1]
    rope = q_pe.shape[-1]
    q_out[:, :heads, :nope].copy_(ql_nope)
    if rope:
        q_out[:, :heads, nope : nope + rope].copy_(q_pe)


def patch_empty_ops() -> bool:
    if not iluvatar_active():
        return False
    value = os.environ.get(_SWITCH, "1").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return False
    global _LIB
    _LIB = torch.library.Library("_C_cache_ops", "FRAGMENT")
    if not hasattr(torch.ops._C_cache_ops, "concat_mla_q"):
        _LIB.define("concat_mla_q(Tensor ql_nope, Tensor q_pe, Tensor! q_out) -> ()")
    _LIB.impl("concat_mla_q", _concat_mla_q, "CUDA")
    return True


patch_empty_ops()
