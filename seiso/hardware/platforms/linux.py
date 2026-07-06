"""Linux system probing."""

from __future__ import annotations

import platform

from seiso.hardware.platforms import common


def disk_usage_root() -> str:
    return "/"


def ram_gb() -> float:
    psutil_val = common.ram_gb_from_psutil()
    if psutil_val is not None:
        return psutil_val
    return common.ram_gb_from_sysconf() or 0.0


def cpu_brand() -> str:
    brand = common.cpu_brand_from_cpuinfo()
    if brand:
        return brand
    return common.cpu_brand_from_platform(platform.processor(), platform.machine())


def cpu_cores() -> int:
    return common.cpu_cores()
