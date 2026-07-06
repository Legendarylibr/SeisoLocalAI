"""Windows system probing."""

from __future__ import annotations

import os
import platform

from seiso.hardware.platforms import common


def disk_usage_root() -> str:
    return os.environ.get("SYSTEMDRIVE", "C:") + "\\"


def ram_gb() -> float:
    return common.ram_gb_from_psutil() or 0.0


def cpu_brand() -> str:
    brand = common.cpu_brand_from_cpuinfo()
    if brand:
        return brand
    return common.cpu_brand_from_platform(platform.processor(), platform.machine())


def cpu_cores() -> int:
    return common.cpu_cores()
