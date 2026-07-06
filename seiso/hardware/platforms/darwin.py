"""macOS / Darwin system probing."""

from __future__ import annotations

import platform

from seiso.hardware.platforms import common


def disk_usage_root() -> str:
    return "/"


def ram_gb() -> float:
    return common.ram_gb_from_psutil() or 0.0


def cpu_brand() -> str:
    brand = common.cpu_brand_from_cpuinfo()
    if brand:
        return brand
    machine = platform.machine()
    if machine in ("arm64", "aarch64"):
        return "Apple Silicon"
    return common.cpu_brand_from_platform(platform.processor(), machine)


def cpu_cores() -> int:
    return common.cpu_cores()
