"""OS-specific system probing — disk, RAM, CPU (not GPU backends)."""

from __future__ import annotations

import platform
from types import ModuleType

__all__ = [
    "cpu_brand",
    "cpu_cores",
    "disk_usage_root",
    "probe_for",
    "ram_gb",
]


def probe_for(system: str | None = None) -> ModuleType:
    """Return the platform probe module for *system* (defaults to current OS)."""
    name = (system or platform.system()).lower()
    if name == "windows":
        from seiso.hardware.platforms import windows

        return windows
    if name == "darwin":
        from seiso.hardware.platforms import darwin

        return darwin
    from seiso.hardware.platforms import linux

    return linux


def disk_usage_root(system: str | None = None) -> str:
    return probe_for(system).disk_usage_root()


def ram_gb(system: str | None = None) -> float:
    return probe_for(system).ram_gb()


def cpu_brand(system: str | None = None) -> str:
    return probe_for(system).cpu_brand()


def cpu_cores(system: str | None = None) -> int:
    return probe_for(system).cpu_cores()
