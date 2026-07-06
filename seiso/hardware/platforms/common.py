"""Shared OS-agnostic system probing helpers."""

from __future__ import annotations

import os

from seiso.hardware.probes.common import sanitize_hardware_label


def cpu_cores() -> int:
    return os.cpu_count() or 1


def ram_gb_from_psutil() -> float | None:
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        return None


def ram_gb_from_sysconf() -> float | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        return round((page_size * phys_pages) / (1024**3), 1)
    except (AttributeError, OSError, ValueError):
        return None


def cpu_brand_from_cpuinfo() -> str | None:
    try:
        import cpuinfo  # type: ignore

        brand = cpuinfo.get_cpu_info().get("brand_raw", "")
        if brand:
            return sanitize_hardware_label(brand)
    except ImportError:
        pass
    return None


def cpu_brand_from_platform(processor: str, machine: str) -> str:
    return sanitize_hardware_label(processor or machine)
