# GLM-5.3-Flash on MetaX C550 (TP16, 2 nodes): startup fixes and remaining blocker

## Summary

Bringing GLM-5.3-Flash up on 2×8 MetaX C550 hit four independent problems. This branch
carries the two that belong in `vllm-plugin-FL`:

1. **`PlatformFL.update_block_size_for_backend`** — keeps `block_size` compatible with
   GLM5-Next's compressed sparse indexer after the attention backend has chosen its
   preferred value. Without it, a `block_size` that is not a multiple of
   `index_kpool * indexer_page` aborts startup with
   `ValueError: GLM5 indexer storage N must align to M`.
2. **`MCCLLibrary.ncclCommAbort`** — MCCL does not export that symbol, so pynccl's
   error-teardown path raised `AttributeError` and hid the real failure.

Two further problems are **not** in this repository and are documented here so they can be
tracked: a MACA driver behaviour that stalls CUDA stream creation when one GPU accumulates
too many contexts, and a FlagGems kernel that needs a NVIDIA-only MMA IR API.

## Environment

| | |
|---|---|
| Accelerators | 2 nodes × 8× MetaX C550 (64 GB), TP16 |
| Stack | vLLM 0.24.0+flagos, torch 2.10.0+metax3.8.1.0, FlagGems / FlagGems-vllm / vllm-plugin-FL |
| Launch | `--distributed-executor-backend mp --nnodes 2`, `--enforce-eager`, `MODE=eager` |
| Model | GLM-5.3-Flash-BF16 (45 layers, 288 routed experts, MTP head) |

## Problem 1 — startup aborts on the compressed indexer layout

`glm5/kpool.py` derives the indexer's physical layout from
`storage_block_size = block_size // index_kpool` and requires

```
storage_block_size % glm5_indexer_page_alignment(capability) == 0
```

where the page is 64 on `capability.major < 10` (DeepGEMM on SM90 only accepts 64-entry
pages) and 32 otherwise. With the model's `index_kpool = 4` and the 64-token block that the
FlagGems MLA backend prefers:

```
storage = 64 // 4 = 16        16 % 64 = 16      -> ValueError: GLM5 indexer storage 16 must align to 64
```

`glm5/runtime.py` and `glm5/cache.py` already intend to round `block_size` up to
`kpool * page`, but they run earlier; in the worker process
`Platform.update_block_size_for_backend()` (`multiproc_executor.py`) runs afterwards and
overwrites `block_size` with the backend's preferred value, discarding that alignment.

### Fix

`PlatformFL.update_block_size_for_backend()` calls `super()` (keeping the backend
preference and the hybrid/mamba alignment) and then re-applies the indexer constraint,
mirroring what `glm5/cache.py` does with `mamba_page_size_padded`.

It is a no-op when the backend's choice is already valid, which is the case in the default
hybrid configuration we tested (`block_size` becomes 320 through the mamba/attention page
alignment, which is itself a multiple of 256). It does act for configurations where the
backend value is not aligned — e.g. a small explicit `--block-size`, or a non-hybrid
dispatch path — where startup previously aborted.

## Problem 2 — `AttributeError` masks the real error

```
AttributeError: 'MCCLLibrary' object has no attribute 'ncclCommAbort'
```

`pynccl.PyNcclCommunicator.__del__` calls `self.nccl.ncclCommAbort(self.comm)` during
teardown. `MCCLLibrary` does not define it and MCCL does not export `mcclCommAbort`, so the
attribute error replaced the original failure. During this investigation it appeared 16
times and repeatedly obscured the true root cause.

### Fix

`ncclCommAbort` degrades to `mcclCommDestroy`; if neither symbol exists it returns quietly.

## Problem 3 — not in this repo: MACA stalls stream creation on a context-heavy GPU

`py-spy --native` on the stuck worker:

```
mxr::Semaphore::timedWait (semaphore.cpp:88)
mxr::HostQueue::start (mx_commandqueue.cpp:120)
mcr::Stream::Create (mc_stream.cpp:483)
mcStreamCreateWithPriority -> c10::cuda::initSingleStream
c10::cuda::getStreamFromPool
c10d::ProcessGroupNCCL::initNCCLComm            <- first collective of the process
```

This is before any MCCL communicator is created. `mx-smi` shows why: every worker in the
node leaves an 18 MiB context on GPU 0, so GPU 0 carries 8 contexts while GPUs 1–7 carry
one each. Under that condition the driver fails to create a host queue:

```
[MXKW][E]queues.c:844  mxkwCreateQueueBlock: ioctl create queue block timeout (type:21)
MXCD.B2B00.D0.RINGBUF.ERROR  create ringbuf failed, -110          (2b:00.0 = GPU 0)
```

Minimal reproduction: with two holder processes on GPU 0 the *second* process still
succeeds (`Retried ioctl create queue block succeed`), the third blocks forever in
`Semaphore::timedWait`. A bare `stream + matmul` probe passes on all 8 GPUs, so the hardware
is fine.

**Effective mitigation (outside this repo):** pin the device in
`vllm/v1/executor/multiproc_executor.py::worker_main()` right after the signal handlers,
before any CUDA call — `torch.cuda.set_device(kwargs["local_rank"])`. With that, startup
reaches `Application startup complete`; without it, the run stalls in `profile_run`
(`shm_broadcast: No available shared memory broadcast block found in 300 seconds`).
`torch.cuda.is_initialized()` is already `True` and `current_device()` already `0` at that
point, so setting the device there also stops the empty GPU-0 contexts from accumulating.
We suggest making this behaviour the default for non-NVIDIA platforms.

## Problem 4 — not in this repo: `flash_mla_sparse_fwd` needs an NVIDIA-only MMA API

With startup fixed, the first request fails compiling the sparse-MLA kernel:

```
AttributeError: 'triton._C.libtriton.ir.builder' object has no attribute
                'make_nv_mma_shared_encoding_attr'
  -> flaggems_vllm/ops/flash_mla_sparse_fwd, via flagtree metax backend make_llir
```

A related Triton lowering gap was fixable locally: `flaggems_vllm/ops/causal_conv1d.py`
uses `tl.gather(..., axis=1)`, which lowers to `maca.shfl.sync` — an op the MACA backend
does not implement (`custom op 'maca.shfl.sync' is unknown`). Replacing that 4×4 register
rotation with an equivalent `where` + `sum` removing the shuffle makes it compile. Both
changes belong to FlagGems-vllm and are not part of this branch.

## Verification

Each configuration was run end to end on the same 2-node TP16 setup.

| Configuration | Startup | Notes |
|---|---|---|
| baseline (before this branch) | ❌ stall in `profile_run` | first collective never returns |
| this branch only | ❌ stall in `profile_run` | indexer constraint satisfied, stream creation still blocks |
| device pinned in `worker_main` | ✅ `Application startup complete` | KV cache 6,389,760 tokens |
| device pinned + `--block-size 512` | ✅ `Application startup complete` | same KV cache size |

The indexer alignment failure itself was reproduced independently with
`GLM53_NO_HYBRID=1` (which bypasses the hybrid page alignment that happens to satisfy the
constraint), showing `ValueError: GLM5 indexer storage 16 must align to 64` at
`glm5/kpool.py:291`.

## Status

Day-0 smoke check **passes** on 2 nodes × 8× MetaX C550 (TP16, eager):

```
问:中国的首都是哪个城市呢？答:应该是  ->  北京。北京。北京是中国的首都
```

8/8 runs consistent, plus `What is the capital of France?` → ` Paris`.

Three things were needed, in this order:

1. **This branch** — keep the GLM5 compressed-indexer block size valid, and stop
   `MCCLLibrary` from masking real errors behind an `AttributeError`.
2. **`tonyh168/FlagGems-vllm`, branch `fix/glm5-metax-causal-conv-tile`** —
   replace `tl.gather(axis=1)` in `causal_conv1d` (it lowers to `maca.shfl.sync`,
   which the MACA backend does not implement) and add sparse-MLA tile presets that
   fit in 64 KB of shared memory instead of requiring ~160 KB.
3. **Device pinning in `vllm/v1/executor/multiproc_executor.py::worker_main()`**
   (vLLM core, not in this repo) — `torch.cuda.set_device(local_rank)` before any
   CUDA call, so GPU 0 stops accumulating one empty context per worker.

### Remaining caveat

The model answers the day-0 prompt but degrades after a few tokens
(`1+1` → `1+1等于2,1+1等于3,`; self-introduction turns into an enumerated loop).
That is an accuracy question rather than a startup/compile one, and it is not
addressed here.

### Suggestions

- Make the early `set_device` the non-NVIDIA default in vLLM's `worker_main`.
- `ncclCommAbort` fallback where MCCL wrappers are registered (done in this branch).
- FlagGems gates the flashmla / chunk_gdr TLE switches on a hardcoded default of
  `"1"` instead of `runtime.device.info.tle_enabled`, which already reports `False`
  on MetaX. `top_k_per_row_*` consults that probe correctly; the flashmla and
  chunk-gdr families do not. `FLAGGEMS_FLASHMLA_SPARSE_TLE=0` is worked around
  locally in the launch script.

## Branch / commit

| Repository | Branch | Commit | Contents |
|---|---|---|---|
| `tonyh168/vllm-plugin-FL` | `fix/glm5-metax-indexer-block-size-mccl-abort` | see this branch | `vllm_fl/platform.py` indexer block-size alignment, `pynccl_wrapper.py` `ncclCommAbort` |
| `tonyh168/vllm-plugin-FL` | `fix/glm5-metax-indexer-block-size-mccl-abort-day0` | `3da1de1` | same fixes on top of the full day-0 tree (+80 commits / 68 files) |
| `tonyh168/FlagGems-vllm` | `fix/glm5-metax-causal-conv-tile` | `1c40d46` | `causal_conv1d.py` gather replacement, `flashmla_sparse.py` tile presets, its own `pr_content.md` |

Base for both repositories is upstream `main` (`flagos-ai`), so the diffs stay
reviewable: this branch is +3 files / +231 lines, the FlagGems branch is
+3 files / +173 lines / -6 lines.

