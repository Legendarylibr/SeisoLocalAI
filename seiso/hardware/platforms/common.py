"""Shared OS-agnostic system probing helpers."""

from __future__ import annotations

import os

from seiso.hardware.probes.common import sanitize_hardware_label

# platform.processor()/machine() often return only the arch on native Linux
# (e.g. "x86_64") when py-cpuinfo is not installed — not a useful brand string.
_ARCH_ONLY_LABELS = frozenset(
    {
        "x86_64",
        "amd64",
        "x64",
        "i386",
        "i686",
        "x86",
        "aarch64",
        "arm64",
        "armv7l",
        "armv8l",
        "ppc64le",
        "ppc64",
        "riscv64",
        "mips64",
        "unknown",
    }
)


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
        if brand and not _is_arch_only_label(brand):
            return sanitize_hardware_label(brand)
    except ImportError:
        pass
    return None


def cpu_brand_from_proc_cpuinfo(path: str = "/proc/cpuinfo") -> str | None:
    """Read model name from Linux /proc/cpuinfo (no third-party deps).

    On native Linux, ``platform.processor()`` often returns only the arch
    (``x86_64``) when py-cpuinfo is absent — /proc is the ground truth.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                lower = line.lower()
                if not (
                    lower.startswith("model name")
                    or lower.startswith("hardware")
                    or lower.startswith("cpu model")
                ):
                    continue
                if ":" not in line:
                    continue
                brand = line.split(":", 1)[1].strip()
                if brand and not _is_arch_only_label(brand):
                    return sanitize_hardware_label(brand)
    except (OSError, FileNotFoundError):
        pass
    return None


def _is_arch_only_label(label: str) -> bool:
    return label.strip().lower() in _ARCH_ONLY_LABELS


def cpu_brand_from_platform(processor: str, machine: str) -> str:
    for candidate in (processor, machine):
        text = (candidate or "").strip()
        if text and not _is_arch_only_label(text):
            return sanitize_hardware_label(text)
    # Last resort — still better than an empty string for the UI.
    return sanitize_hardware_label((processor or machine or "CPU").strip() or "CPU")
