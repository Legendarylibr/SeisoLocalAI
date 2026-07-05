"""Local hardware detection and profile caching — no telemetry, no external calls."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from typing import Any

from seiso.hardware.gpus import sanitize_hardware_label
from seiso.hardware.tiers import (
    TIER_LABELS,
    classify_tier,
    effective_budget_mb,
    vram_headroom_mb,
)
from seiso.hardware.training import preferred_inference_backend, training_defaults
from seiso.models.loader import detect_backend

_PROFILE_TTL_S = 30.0
_METRICS_TTL_S = 1.5
_profile_cache: dict[str, Any] | None = None
_profile_cache_ts: float = 0.0
_metrics_cache: dict[str, Any] | None = None
_metrics_cache_ts: float = 0.0
_cpu_percent_primed = False


def _disk_usage_root() -> str:
    """Filesystem root used for free-space reporting (OS-appropriate)."""
    if platform.system().lower() == "windows":
        return os.environ.get("SYSTEMDRIVE", "C:") + "\\"
    return "/"


def _ram_gb() -> float:
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        pass
    try:
        page_size = __import__("os").sysconf("SC_PAGE_SIZE")
        phys_pages = __import__("os").sysconf("SC_PHYS_PAGES")
        return round((page_size * phys_pages) / (1024**3), 1)
    except (AttributeError, OSError, ValueError):
        return 0.0


def _cpu_brand() -> str:
    try:
        import cpuinfo  # type: ignore

        brand = cpuinfo.get_cpu_info().get("brand_raw", "")
        if brand:
            return sanitize_hardware_label(brand)
    except ImportError:
        pass
    proc = platform.processor() or platform.machine()
    if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        return "Apple Silicon"
    return sanitize_hardware_label(proc)


def _cpu_cores() -> int:
    return __import__("os").cpu_count() or 1


def _nvidia_smi_metrics() -> dict[int, dict[str, float]]:
    from seiso.security.nvidia_boundary import resolve_nvidia_smi_executable

    out: dict[int, dict[str, float]] = {}
    smi = resolve_nvidia_smi_executable()
    if not smi:
        return out
    try:
        result = subprocess.run(
            [
                smi,
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.returncode != 0:
            return out
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            idx = int(parts[0])
            out[idx] = {
                "utilization_pct": (
                    float(parts[1]) if parts[1] not in ("[N/A]", "N/A") else 0.0
                ),
                "vram_used_mb": float(parts[2]),
                "vram_total_mb": float(parts[3]),
                "temperature_c": (
                    float(parts[4]) if parts[4] not in ("[N/A]", "N/A") else 0.0
                ),
            }
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return out


def detect_gpus() -> list[dict[str, Any]]:
    from seiso.hardware.gpus import enumerate_gpus

    gpus = [dict(gpu) for gpu in enumerate_gpus()]
    smi = _nvidia_smi_metrics()
    for gpu in gpus:
        idx = gpu.get("index", 0)
        if idx in smi:
            gpu.update({k: v for k, v in smi[idx].items() if v is not None})
    return gpus


def live_metrics() -> dict[str, Any]:
    """Snapshot of CPU/RAM/GPU — aggregated locally, never exported."""
    global _metrics_cache, _metrics_cache_ts, _cpu_percent_primed

    now = time.time()
    if _metrics_cache is not None and now - _metrics_cache_ts < _METRICS_TTL_S:
        return _metrics_cache

    cpu_util: float | None = None
    cpu_temp: float | None = None
    ram_used_pct = 0.0

    try:
        import psutil  # type: ignore

        cpu_util = round(
            psutil.cpu_percent(interval=None if _cpu_percent_primed else 0.05),
            1,
        )
        _cpu_percent_primed = True
        ram_used_pct = round(psutil.virtual_memory().percent, 1)
        sensors_temperatures = getattr(psutil, "sensors_temperatures", None)
        temps = sensors_temperatures() if sensors_temperatures else {}
        for key in ("coretemp", "cpu_thermal", "TC0P", "TH0x"):
            if key in temps and temps[key]:
                cpu_temp = round(temps[key][0].current, 1)
                break
    except (ImportError, AttributeError):
        pass

    result = {
        "cpu_util_pct": cpu_util,
        "cpu_temp_c": cpu_temp,
        "ram_used_pct": ram_used_pct,
        "gpus": detect_gpus(),
        "local_only": True,
        "ts": now,
    }
    _metrics_cache = result
    _metrics_cache_ts = now
    return result


def enrich_profile_base(profile: dict[str, Any]) -> dict[str, Any]:
    """Add tier, headroom, and training defaults without Forge catalog lookups."""
    tier = classify_tier(profile)
    headroom = vram_headroom_mb(profile)
    return {
        **profile,
        "tier": tier.value,
        "tier_label": TIER_LABELS[tier],
        "effective_vram_mb": effective_budget_mb(profile),
        "vram_headroom_mb": headroom,
        "preferred_inference_backend": preferred_inference_backend(profile),
        "training_defaults": training_defaults(profile),
    }


def hardware_profile(*, force_refresh: bool = False) -> dict[str, Any]:
    """Local hardware snapshot with tier, headroom, and training defaults (cached 30s).

    Does not include Hub catalog recommendations — use ``forge.services.hardware.hardware_profile``
    when those fields are required.
    """
    global _profile_cache, _profile_cache_ts

    now = time.time()
    if (
        not force_refresh
        and _profile_cache is not None
        and now - _profile_cache_ts < _PROFILE_TTL_S
    ):
        return _profile_cache

    backend = detect_backend()
    gpus = detect_gpus()
    ram = _ram_gb()
    try:
        disk_free = shutil.disk_usage(_disk_usage_root()).free // (1024**3)
    except (OSError, FileNotFoundError):
        disk_free = 0

    cuda_runtime = False
    try:
        import torch

        cuda_runtime = torch.cuda.is_available()
    except ImportError:
        pass

    profile = {
        "platform": platform.system().lower(),
        "arch": platform.machine(),
        "backend": backend.value,
        "cuda_runtime": cuda_runtime,
        "cpu_cores": _cpu_cores(),
        "cpu_brand": _cpu_brand(),
        "ram_gb": ram,
        "disk_free_gb": disk_free,
        "gpus": gpus,
        "local_only": True,
        "privacy": "Metrics are read locally and never sent off this machine.",
    }
    _profile_cache = enrich_profile_base(profile)
    _profile_cache_ts = now
    return _profile_cache
