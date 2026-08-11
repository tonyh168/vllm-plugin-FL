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

import glob
import os

import vllm
import vllm.utils.nccl


def find_inccl_library() -> str:
    """
    Locate Iluvatar CoreX libnccl.so. Iluvatar's NCCL uses the `pnccl`
    symbol prefix internally; this is still the same .so file.

    Search order:
    1. VLLM_NCCL_SO_PATH env var (user override)
    2. LD_LIBRARY_PATH directories
    3. Well-known CoreX install paths (/usr/local/corex*/lib64/libnccl.so)
    """
    so_file = os.environ.get("VLLM_NCCL_SO_PATH")
    if so_file:
        return so_file

    for dir_ in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        candidate = os.path.join(dir_, "libnccl.so")
        if os.path.isfile(candidate):
            return candidate

    # Fall back to any installed CoreX version, newest first.
    candidates = sorted(
        glob.glob("/usr/local/corex*/lib64/libnccl.so"), reverse=True
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "Iluvatar libnccl.so not found. Set VLLM_NCCL_SO_PATH or ensure "
        "the CoreX SDK is installed under /usr/local/corex*/lib64/."
    )


vllm.utils.nccl.find_nccl_library = find_inccl_library
