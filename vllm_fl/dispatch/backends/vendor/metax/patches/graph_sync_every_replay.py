# Copyright (c) 2026 BAAI. All rights reserved.

"""Diagnostic: synchronise after *every* CUDA graph replay.

Why
---
Minimal repro (``repro_mccl_graph_deadlock.py``) narrows the MCCL 2.16.5
graph deadlock on MetaX C550 to one sequence::

    g.replay()
    dist.all_gather_into_tensor(y, x)   # eager, same communicator, no sync
    g.replay()                          # <-- hangs
    torch.cuda.synchronize()

Every other isolated combination passes (single replay, replay+eager, graph with
16 collectives, two communicators inside one graph, N replays back to back), so
the trigger is specifically **an eager collective sitting between two replays**.
MCCL's own text: "streams given to a communicator within a MCCL group must
either be all uncaptured or all captured by the same graph".

In vLLM PIECEWISE, the ops in ``splitting_ops`` (attention, kv-cache update)
are executed **eagerly between** the captured pieces of every decode step, so a
single step already has the deadlocking shape before the logits gather is even
reached. ``graph_eager_sync.py`` only covers the logits gather, which is not
enough.

What this does
--------------
Append ``torch.cuda.current_stream().synchronize()`` after each
``torch.cuda.CUDAGraph.replay()``. That is a blunt instrument — it defeats most
of the launch-overhead win — but it answers one question conclusively: *is
graph/eager interleaving the only thing standing between us and working
cudagraphs?* If PIECEWISE runs clean with this on, the upstream fix belongs in
MCCL (or in vLLM's piecewise scheduler), not in our integration.

Off by default: ``VLLM_FL_METAX_GRAPH_SYNC_EVERY_REPLAY=1`` turns it on.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_APPLIED = False


def _install() -> bool:
    global _APPLIED
    if _APPLIED:
        return True
    if os.environ.get("VLLM_FL_METAX_GRAPH_SYNC_EVERY_REPLAY", "0") not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False

    original = torch.cuda.CUDAGraph.replay
    if getattr(original, "_fl_metax_synced", False):
        _APPLIED = True
        return True

    def replay_then_sync(self):
        original(self)
        torch.cuda.current_stream().synchronize()

    replay_then_sync._fl_metax_synced = True  # type: ignore[attr-defined]
    replay_then_sync._fl_original = original  # type: ignore[attr-defined]
    torch.cuda.CUDAGraph.replay = replay_then_sync
    _APPLIED = True
    logger.info(
        "graph-sync-every-replay: torch.cuda.CUDAGraph.replay now synchronises "
        "the stream after each replay (diagnostic; expect a slowdown)"
    )
    return True


_install()
