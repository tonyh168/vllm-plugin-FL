from . import accelerator_compat
from . import functorch_config_patch
from . import fix_standalone_compile
from . import pynccl_wrapper
from . import cuda_wrapper
from . import utils_patch
from . import chunk_delta_h
from . import topk_topp_sampler
from . import gdn_linear_attn  # noqa: F401 — register MacaGatedDeltaNetAttention
from . import int8_quant  # noqa: F401 — portable dynamic INT8 quantization for W8A8
from . import int8_linear  # noqa: F401 — keep the INT8 Triton scaled-mm opaque to dynamo
from . import graph_eager_sync  # noqa: F401 — MCCL: sync after the eager logits all-gather
from . import mla_safe_tiles  # noqa: F401 — fit sparse MLA prefill into 64 KB smem

# --------------------------------------------------
# MetaX C550 does not support third-party Triton kernels (Triton upgrade required).
# Disable them so FLA decode ops (fused_recurrent_gated_delta_rule etc.) fall back
# to the non-Triton path handled by mcoplib, producing correct output.
# TODO: remove when MetaX Triton support is available.
import vllm.utils.import_utils as iu
iu.has_triton_kernels = lambda: False
