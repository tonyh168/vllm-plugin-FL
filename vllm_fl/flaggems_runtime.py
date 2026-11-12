# Copyright (c) 2026 BAAI. All rights reserved.
"""Process-wide FlagGems initialization after policy resolution."""
from dataclasses import dataclass
from threading import RLock


@dataclass(frozen=True)
class FlagGemsConfig:
    whitelist: tuple[str, ...] | None
    blacklist: tuple[str, ...] | None


_STATE: FlagGemsConfig | None = None
_FAILED = False
_LOCK = RLock()


def configure_flaggems(
    enable_flaggems, *, use_flaggems=True, whitelist=None, blacklist=None
) -> None:
    global _STATE, _FAILED
    with _LOCK:
        if _FAILED:
            raise RuntimeError(
                "FlagGems initialization previously failed; restart the process"
            )
        if not use_flaggems:
            if _STATE is not None:
                raise RuntimeError(
                    "FlagGems configuration is process-lifetime; restart to disable it"
                )
            return
        config = FlagGemsConfig(
            tuple(sorted(whitelist)) if whitelist is not None else None,
            tuple(sorted(blacklist)) if blacklist is not None else None,
        )
        if _STATE is not None:
            if config != _STATE:
                raise RuntimeError(
                    "FlagGems configuration is process-lifetime; restart to change it"
                )
            return
        try:
            enable_flaggems(None)
            _STATE = config
        except Exception:
            _FAILED = True
            raise


def should_record_oplist(rank):
    """Keep recording compatible by default; allow disabling diagnostic logs."""
    import os

    value = os.environ.get("VLLM_FL_RECORD_OPLIST", "1")
    if value not in ("0", "1"):
        raise ValueError("VLLM_FL_RECORD_OPLIST must be 0 or 1")
    return rank == 0 and value == "1"


def emit_dispatch_receipt(*, rank, local_rank=None, whitelist=None, blacklist=None):
    """Emit the actual registration once at startup, never during inference."""
    import json
    import os

    from vllm.platforms import current_platform
    import vllm_fl.envs as fl_envs

    enabled = bool(fl_envs.USE_FLAGGEMS)
    if enabled:
        import flag_gems
        names = {str(name) for name in flag_gems.all_registered_ops()}
    else:
        names = set()
    append = os.environ.get("VLLM_FL_FLAGOS_BLACKLIST_APPEND", "")
    selected = {name.strip() for name in append.split(",") if name.strip()}
    # Do not dump the environment, touch tensors or install a forward hook.
    print("FLAGOS_DISPATCH_RECEIPT " + json.dumps({
        "rank": rank,
        "oplist_recording_enabled": should_record_oplist(rank),
        "local_rank": local_rank,
        "vendor": getattr(current_platform, "vendor_name", None),
        "global_flaggems_enabled": enabled,
        "configured_append": append,
        "effective_whitelist": whitelist,
        "effective_blacklist": blacklist,
        "registered_ops_count": len(names),
        "selected_names_still_registered": sorted(names & selected),
    }, sort_keys=True), flush=True)
