"""Host platform detection shared across kernels and security."""

from __future__ import annotations

import os
from pathlib import Path


def detect_wsl2() -> bool:
    """True when running inside WSL2."""
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        version = Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return "microsoft" in version or "wsl2" in version
