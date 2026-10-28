# Copyright (c) 2026 BAAI. All rights reserved.

"""Serialise the eager logits all-gather against the next CUDA graph replay.

Why
---
On MetaX C550 + MCCL 2.16.5, enqueueing a CUDA graph replay while an *eager*
collective on the same communicator is still in flight deadlocks: every rank
blocks inside the replay until the watchdog fires. Minimal repro (no vLLM, no
model) lives next to the Hy4-1022 notes as ``repro_mccl_graph_deadlock.py``::

    g.replay()
    dist.all_gather_into_tensor(y, x)   # same communicator, no sync
    g.replay()                          # <-- hangs
    torch.cuda.synchronize()

Adding ``synchronize()`` between the eager collective and the next replay makes
the very same sequence pass, and MCCL's own error text says why:

    Streams given to a communicator within a MCCL group must either be all
    uncaptured or all captured by the same graph.

vLLM always performs the logits all-gather **outside** the compiled graph
(``LogitsProcessor._gather_logits`` is not ``support_torch_compile``-decorated),
so every decode step is ``graph replay -> eager all-gather -> graph replay`` --
exactly the deadlocking sequence. That is why both ``PIECEWISE`` and
``FULL_DECODE_ONLY`` hang on the first request while ``cudagraph_mode=NONE``
works.

What this patch does
--------------------
After ``_gather_logits`` returns, synchronise the current stream once, so the
eager collective is complete before the next graph replay is enqueued.

Cost: one stream sync per decode step. At the speeds this stack runs
(~0.6 s/step) that is noise; it is *not* a general-purpose setting for a fast
accelerator, hence the opt-out below.

Knobs
-----
``VLLM_FL_METAX_GRAPH_EAGER_SYNC``
    ``auto`` (default) — only when a non-``NONE`` cudagraph mode is active.
    ``1``              — always (for experiments).
    ``0``              — never.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_APPLIED = False


def _enabled() -> bool:
    raw = os.environ.get("VLLM_FL_METAX_GRAPH_EAGER_SYNC", "auto").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    # auto: only when some cudagraph mode is active
    try:
        from vllm.config import CUDAGraphMode, get_current_vllm_config_or_none

        cfg = get_current_vllm_config_or_none()
        if cfg is None:
            return False
        mode = cfg.compilation_config.cudagraph_mode
        return mode is not None and mode != CUDAGraphMode.NONE
    except Exception:  # noqa: BLE001 - config not ready / older torch
        return False


def _install() -> bool:
    global _APPLIED
    if _APPLIED:
        return True
    try:
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
    except Exception as exc:  # noqa: BLE001
        logger.debug("graph-eager-sync: LogitsProcessor unavailable (%s)", exc)
        return False

    original = LogitsProcessor._gather_logits
    if getattr(original, "_fl_metax_synced", False):
        _APPLIED = True
        return True

    def _gather_logits_synced(self, logits):
        out = original(self, logits)
        if _enabled():
            # Wait for the all-gather to retire before the next graph replay can
            # be enqueued (see module docstring for the MCCL deadlock).
            torch.cuda.current_stream().synchronize()
        return out

    _gather_logits_synced._fl_metax_synced = True  # type: ignore[attr-defined]
    _gather_logits_synced._fl_original = original  # type: ignore[attr-defined]
    LogitsProcessor._gather_logits = _gather_logits_synced
    _APPLIED = True
    logger.info(
        "graph-eager-sync: LogitsProcessor._gather_logits now synchronises the "
        "stream after the eager logits all-gather (VLLM_FL_METAX_GRAPH_EAGER_SYNC=%s)",
        os.environ.get("VLLM_FL_METAX_GRAPH_EAGER_SYNC", "auto"),
    )
    return True


_install()
