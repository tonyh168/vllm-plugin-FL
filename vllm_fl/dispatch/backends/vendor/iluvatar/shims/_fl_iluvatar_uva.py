"""UVA view of pinned CPU tensors on Iluvatar with the empty vLLM wheel.

HY4-Flash keeps its Engram tables in pinned host memory and reads them from
Triton kernels through Unified Virtual Addressing.  ``vllm._C`` (which owns
``get_cuda_view_from_cpu_tensor``) is not built in the empty wheel, and
``PlatformFL`` reports ``is_cuda_alike() == False`` so vLLM's Python helper
refuses before reaching the op.

Both gaps are closed here: the ``_C`` op is bound to a DLPack alias of the
pinned host buffer (verified on BI-V150), and the Python helper is routed to
it on Iluvatar.

Kill switch: ``FL_ILUVATAR_UVA=0``.
"""

import ctypes
import os
import sys

import torch
from torch.utils.dlpack import from_dlpack

from ._fl_iluvatar_gate import iluvatar_active

_SWITCH = "FL_ILUVATAR_UVA"


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


_DTYPE = {
    torch.float32: (2, 32),
    torch.bfloat16: (4, 16),  # kDLBfloat
    torch.float16: (2, 16),
    torch.int64: (0, 64),
    torch.int32: (0, 32),
    torch.int16: (0, 16),
    torch.int8: (0, 8),
    torch.uint8: (1, 8),
    torch.bool: (6, 8),
}

# Keep the ctypes structs/arrays alive for the lifetime of the alias.
_KEEP: dict[int, tuple] = {}


def get_uva_view(cpu_tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(cpu_tensor, torch.Tensor):
        raise TypeError("get_cuda_view_from_cpu_tensor expects a tensor")
    if cpu_tensor.device.type != "cpu":
        raise ValueError(
            f"get_cuda_view_from_cpu_tensor expects a CPU tensor, got {cpu_tensor.device}"
        )
    if not cpu_tensor.is_pinned():
        raise ValueError("get_cuda_view_from_cpu_tensor requires pinned host memory")
    fp8 = cpu_tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    if not fp8 and cpu_tensor.dtype not in _DTYPE:
        raise TypeError(f"unsupported dtype for UVA alias: {cpu_tensor.dtype}")

    shape = (ctypes.c_int64 * cpu_tensor.dim())(*cpu_tensor.shape)
    strides = (ctypes.c_int64 * cpu_tensor.dim())(*cpu_tensor.stride())
    code, bits = (1, 8) if fp8 else _DTYPE[cpu_tensor.dtype]

    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(cpu_tensor.data_ptr())
    managed.dl_tensor.device = _DLDevice(2, torch.cuda.current_device())  # kDLCUDA
    managed.dl_tensor.ndim = cpu_tensor.dim()
    managed.dl_tensor.dtype = _DLDataType(code, bits, 1)
    managed.dl_tensor.shape = shape
    managed.dl_tensor.strides = strides
    managed.dl_tensor.byte_offset = 0
    deleter = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(lambda _p: None)
    managed.deleter = ctypes.cast(deleter, ctypes.c_void_p)

    capsule = ctypes.pythonapi.PyCapsule_New(ctypes.byref(managed), b"dltensor", None)
    view = from_dlpack(capsule)
    if fp8:
        view = view.view(cpu_tensor.dtype)
    _KEEP[view.data_ptr()] = (managed, shape, strides, deleter, cpu_tensor)
    return view


def patch() -> bool:
    if not iluvatar_active():
        return False
    value = os.environ.get(_SWITCH, "1").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return False

    # 1) Make the registered CPU impl of _C::get_cuda_view_from_cpu_tensor use
    #    our alias.  Re-registering is not possible, so swap the helper the
    #    registry's impl resolves from its module globals.
    import vllm_fl.ops._C_ops_registry as registry

    registry._unavailable_cuda_view = get_uva_view

    # 2) vLLM's Python helper refuses on non-cuda-alike platforms; route it to
    #    the op we just made functional.
    import vllm.utils.torch_utils as torch_utils

    original = torch_utils.get_accelerator_view_from_cpu_tensor

    def get_accelerator_view_from_cpu_tensor(cpu_tensor):
        return torch.ops._C.get_cuda_view_from_cpu_tensor(cpu_tensor)

    if not getattr(original, "_iluvatar_uva", False):
        get_accelerator_view_from_cpu_tensor._iluvatar_uva = True
        torch_utils.get_accelerator_view_from_cpu_tensor = (
            get_accelerator_view_from_cpu_tensor
        )
        # Consumers that imported the name by value before this shim ran.
        for mod_name in (
            "vllm_fl.models.hy_v4_flash.nvidia.engram",
            "vllm.models.hy_v4.common.engram",
        ):
            mod = sys.modules.get(mod_name)
            if mod is not None and hasattr(
                mod, "get_accelerator_view_from_cpu_tensor"
            ):
                mod.get_accelerator_view_from_cpu_tensor = (
                    get_accelerator_view_from_cpu_tensor
                )

    print("[vllm_fl][iluvatar] _C.get_cuda_view_from_cpu_tensor -> DLPack UVA")
    return True


patch()
