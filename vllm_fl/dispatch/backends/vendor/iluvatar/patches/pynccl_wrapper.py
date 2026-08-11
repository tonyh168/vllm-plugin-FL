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

# Iluvatar CoreX compatibility patch: replace NCCLLibrary with INCCLLibrary.
#
# Iluvatar's libnccl.so exposes all collective operations under the `pnccl`
# symbol prefix (e.g. pncclGetUniqueId, pncclCommInitRank, ...) rather than
# the standard `nccl` prefix. Calling ncclGetUniqueId on this library returns
# ncclInvalidUsage immediately. This patch re-maps every call to the correct
# pnccl symbol while keeping the public Python interface identical to
# NCCLLibrary so no other vLLM code needs to change.

import ctypes
import platform
from typing import Any

from vllm.distributed.device_communicators.pynccl_wrapper import (
    Function,
    buffer_type,
    cudaStream_t,
    logger,
    ncclComm_t,
    ncclDataType_t,
    ncclRedOp_t,
    ncclResult_t,
    ncclWindow_t,
    ncclUniqueId,
)

from .utils_patch import find_inccl_library


class INCCLLibrary:
    """NCCLLibrary drop-in backed by Iluvatar's pnccl-prefixed symbols."""

    exported_functions = [
        Function("pncclGetErrorString", ctypes.c_char_p, [ncclResult_t]),
        Function("pncclGetVersion", ncclResult_t, [ctypes.POINTER(ctypes.c_int)]),
        Function("pncclGetUniqueId", ncclResult_t, [ctypes.POINTER(ncclUniqueId)]),
        Function(
            "pncclCommInitRank",
            ncclResult_t,
            [ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId, ctypes.c_int],
        ),
        Function(
            "pncclAllReduce",
            ncclResult_t,
            [
                buffer_type, buffer_type, ctypes.c_size_t,
                ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t,
            ],
        ),
        Function(
            "pncclReduce",
            ncclResult_t,
            [
                buffer_type, buffer_type, ctypes.c_size_t,
                ncclDataType_t, ncclRedOp_t, ctypes.c_int, ncclComm_t, cudaStream_t,
            ],
        ),
        Function(
            "pncclAllGather",
            ncclResult_t,
            [
                buffer_type, buffer_type, ctypes.c_size_t,
                ncclDataType_t, ncclComm_t, cudaStream_t,
            ],
        ),
        Function(
            "pncclReduceScatter",
            ncclResult_t,
            [
                buffer_type, buffer_type, ctypes.c_size_t,
                ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t,
            ],
        ),
        Function(
            "pncclSend",
            ncclResult_t,
            [
                buffer_type, ctypes.c_size_t, ncclDataType_t,
                ctypes.c_int, ncclComm_t, cudaStream_t,
            ],
        ),
        Function(
            "pncclRecv",
            ncclResult_t,
            [
                buffer_type, ctypes.c_size_t, ncclDataType_t,
                ctypes.c_int, ncclComm_t, cudaStream_t,
            ],
        ),
        Function(
            "pncclBroadcast",
            ncclResult_t,
            [
                buffer_type, buffer_type, ctypes.c_size_t,
                ncclDataType_t, ctypes.c_int, ncclComm_t, cudaStream_t,
            ],
        ),
        Function("pncclCommDestroy", ncclResult_t, [ncclComm_t]),
        Function("pncclGroupStart", ncclResult_t, []),
        Function("pncclGroupEnd", ncclResult_t, []),
        Function(
            "pncclCommWindowRegister",
            ncclResult_t,
            [
                ncclComm_t, buffer_type, ctypes.c_size_t,
                ctypes.POINTER(ncclWindow_t), ctypes.c_int,
            ],
        ),
        Function("pncclCommWindowDeregister", ncclResult_t, [ncclComm_t, ncclWindow_t]),
    ]

    path_to_library_cache: dict[str, Any] = {}
    path_to_dict_mapping: dict[str, dict[str, Any]] = {}

    def __init__(self, so_file: str | None = None):
        so_file = so_file or find_inccl_library()
        try:
            if so_file not in INCCLLibrary.path_to_library_cache:
                lib = ctypes.CDLL(so_file)
                INCCLLibrary.path_to_library_cache[so_file] = lib
            self.lib = INCCLLibrary.path_to_library_cache[so_file]
        except Exception as e:
            logger.error(
                "Failed to load Iluvatar NCCL library from %s. "
                "Ensure the CoreX SDK is installed and libnccl.so is present. "
                "Platform: %s. Set VLLM_NCCL_SO_PATH to override the path.",
                so_file,
                platform.platform(),
            )
            raise e

        if so_file not in INCCLLibrary.path_to_dict_mapping:
            _funcs: dict[str, Any] = {}
            for func in INCCLLibrary.exported_functions:
                try:
                    f = getattr(self.lib, func.name)
                    f.restype = func.restype
                    f.argtypes = func.argtypes
                    _funcs[func.name] = f
                except AttributeError:
                    if func.name in [
                        "pncclCommWindowRegister",
                        "pncclCommWindowDeregister",
                    ]:
                        logger.warning_once(
                            "Symbol %s not found in %s; "
                            "symmetric memory ops will be unavailable.",
                            func.name, so_file,
                        )
                        continue
                    raise
            INCCLLibrary.path_to_dict_mapping[so_file] = _funcs
        self._funcs = INCCLLibrary.path_to_dict_mapping[so_file]

    # ------------------------------------------------------------------
    # Public interface mirrors NCCLLibrary exactly.
    # ------------------------------------------------------------------

    def ncclGetErrorString(self, result: ncclResult_t) -> str:
        return self._funcs["pncclGetErrorString"](result).decode("utf-8")

    def NCCL_CHECK(self, result: ncclResult_t) -> None:
        if result != 0:
            error_str = self.ncclGetErrorString(result)
            raise RuntimeError(f"NCCL error: {error_str}")

    def ncclGetRawVersion(self) -> int:
        version = ctypes.c_int()
        self.NCCL_CHECK(self._funcs["pncclGetVersion"](ctypes.byref(version)))
        return version.value

    def ncclGetVersion(self) -> str:
        version_str = str(self.ncclGetRawVersion())
        major = version_str[0].lstrip("0")
        minor = version_str[1:3].lstrip("0")
        patch = version_str[3:].lstrip("0")
        return f"{major}.{minor}.{patch}"

    def ncclGetUniqueId(self) -> ncclUniqueId:
        unique_id = ncclUniqueId()
        self.NCCL_CHECK(self._funcs["pncclGetUniqueId"](ctypes.byref(unique_id)))
        return unique_id

    def unique_id_from_bytes(self, data: bytes) -> ncclUniqueId:
        if len(data) != 128:
            raise ValueError(
                f"Expected 128 bytes for ncclUniqueId, got {len(data)} bytes"
            )
        unique_id = ncclUniqueId()
        ctypes.memmove(ctypes.addressof(unique_id.internal), data, 128)
        return unique_id

    def ncclCommInitRank(
        self, world_size: int, unique_id: ncclUniqueId, rank: int
    ) -> ncclComm_t:
        comm = ncclComm_t()
        self.NCCL_CHECK(
            self._funcs["pncclCommInitRank"](
                ctypes.byref(comm), world_size, unique_id, rank
            )
        )
        return comm

    def ncclAllReduce(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["pncclAllReduce"](
                sendbuff, recvbuff, count, datatype, op, comm, stream
            )
        )

    def ncclReduce(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        root: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["pncclReduce"](
                sendbuff, recvbuff, count, datatype, op, root, comm, stream
            )
        )

    def ncclReduceScatter(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["pncclReduceScatter"](
                sendbuff, recvbuff, count, datatype, op, comm, stream
            )
        )

    def ncclAllGather(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["pncclAllGather"](
                sendbuff, recvbuff, count, datatype, comm, stream
            )
        )

    def ncclSend(
        self,
        sendbuff: buffer_type,
        count: int,
        datatype: int,
        dest: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["pncclSend"](sendbuff, count, datatype, dest, comm, stream)
        )

    def ncclRecv(
        self,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        src: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["pncclRecv"](recvbuff, count, datatype, src, comm, stream)
        )

    def ncclBroadcast(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        root: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["pncclBroadcast"](
                sendbuff, recvbuff, count, datatype, root, comm, stream
            )
        )

    def ncclCommDestroy(self, comm: ncclComm_t) -> None:
        self.NCCL_CHECK(self._funcs["pncclCommDestroy"](comm))

    def ncclGroupStart(self) -> None:
        self.NCCL_CHECK(self._funcs["pncclGroupStart"]())

    def ncclGroupEnd(self) -> None:
        self.NCCL_CHECK(self._funcs["pncclGroupEnd"]())

    def ncclCommWindowRegister(
        self, comm: ncclComm_t, buff: buffer_type, size: int, win_flags: int
    ) -> ncclWindow_t:
        window = ncclWindow_t()
        if "pncclCommWindowRegister" in self._funcs:
            self.NCCL_CHECK(
                self._funcs["pncclCommWindowRegister"](
                    comm, buff, size, ctypes.byref(window), win_flags
                )
            )
        return window

    def ncclCommWindowDeregister(self, comm: ncclComm_t, window: ncclWindow_t) -> None:
        if "pncclCommWindowDeregister" in self._funcs:
            self.NCCL_CHECK(
                self._funcs["pncclCommWindowDeregister"](comm, window)
            )


# Patch vLLM's pynccl module to use INCCLLibrary instead of the default
# NCCLLibrary. This must happen before any PyNcclCommunicator is constructed.
from vllm.distributed.device_communicators import pynccl, pynccl_wrapper

pynccl.NCCLLibrary = INCCLLibrary
pynccl_wrapper.NCCLLibrary = INCCLLibrary
