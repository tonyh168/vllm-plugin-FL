"""Relax compressed-tensors capability gates on Iluvatar.

Iluvatar BI-V150 reports compute capability 7.1 while the W8A8 INT8 scheme
declares a minimum of 75.  The plugin installs its own INT8 linear/MoE kernels
(``register_quant_linear``), so the upstream GPU-capability heuristic does not
describe the implementation that actually runs.  Relax it only on Iluvatar.

Kill switch: ``FL_ILUVATAR_QUANT_CAPABILITY=0``.
"""

import os

from ._fl_iluvatar_gate import iluvatar_active

_SWITCH = "FL_ILUVATAR_QUANT_CAPABILITY"


def patch() -> bool:
    if not iluvatar_active():
        return False
    value = os.environ.get(_SWITCH, "1").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return False

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    if getattr(CompressedTensorsConfig, "_iluvatar_capability_relaxed", False):
        return True

    CompressedTensorsConfig._check_scheme_supported = staticmethod(
        lambda min_capability, error=True, match_exact=False: True
    )
    CompressedTensorsConfig._iluvatar_capability_relaxed = True
    print(
        "[vllm_fl][iluvatar] compressed-tensors capability gate relaxed "
        "(device cc 7.1 < scheme minimum)"
    )
    return True


patch()
