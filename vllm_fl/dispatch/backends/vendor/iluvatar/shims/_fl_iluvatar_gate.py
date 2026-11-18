"""Vendor gate for the Iluvatar Hy4-Flash compatibility shims."""

import os


def iluvatar_active() -> bool:
    try:
        from vllm_fl.utils import DeviceInfo

        return DeviceInfo().vendor_name == "iluvatar"
    except Exception:
        return os.environ.get("GEMS_VENDOR", "").strip().lower() == "iluvatar"
