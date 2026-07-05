"""Hardware tier classification and memory budgets."""

from __future__ import annotations

from typing import Any

from seiso.compat import StrEnum
from seiso.models.loader import Backend

FIT_RANK = {"ideal": 4, "good": 3, "tight": 2, "unlikely": 1}


class HardwareTier(StrEnum):
    CPU_ONLY = "cpu_only"
    EDGE = "edge"
    MODEST = "modest"
    CAPABLE = "capable"
    WORKSTATION = "workstation"
    APPLE_UNIFIED = "apple_unified"


TIER_LABELS: dict[HardwareTier, str] = {
    HardwareTier.CPU_ONLY: "CPU only",
    HardwareTier.EDGE: "Edge GPU",
    HardwareTier.MODEST: "Modest GPU",
    HardwareTier.CAPABLE: "Capable GPU",
    HardwareTier.WORKSTATION: "Workstation GPU",
    HardwareTier.APPLE_UNIFIED: "Apple unified memory",
}


def _discrete_gpu_entries(gpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Non-Apple entries from local GPU probes."""
    discrete: list[dict[str, Any]] = []
    for gpu in gpus:
        name = str(gpu.get("name") or "").lower()
        if "apple gpu" in name:
            continue
        discrete.append(gpu)
    return discrete


def classify_tier(profile: dict[str, Any]) -> HardwareTier:
    raw_backend = profile.get("backend", "cpu")
    try:
        backend = Backend(raw_backend)
    except ValueError:
        backend = Backend.TORCH if raw_backend in ("cuda", "rocm") else Backend.CPU
    gpus = profile.get("gpus") or []
    discrete = _discrete_gpu_entries(gpus)
    vram_total = max((g.get("vram_total_mb") or 0) for g in discrete) if discrete else 0

    profile_platform = str(profile.get("platform") or "").lower()
    profile_arch = str(profile.get("arch") or "").lower()
    apple_silicon = profile_platform in {"darwin", "macos"} and profile_arch in {
        "arm64",
        "aarch64",
    }

    if (backend == Backend.MLX or apple_silicon) and not vram_total:
        return HardwareTier.APPLE_UNIFIED
    if not discrete:
        return HardwareTier.CPU_ONLY
    if vram_total <= 0:
        return HardwareTier.EDGE
    if vram_total >= 24000:
        return HardwareTier.WORKSTATION
    if vram_total >= 12000:
        return HardwareTier.CAPABLE
    if vram_total >= 6000:
        return HardwareTier.MODEST
    return HardwareTier.EDGE


def effective_budget_mb(profile: dict[str, Any]) -> int:
    """Memory budget for local inference — full local capacity minus a small reserve."""
    tier = classify_tier(profile)
    gpus = profile.get("gpus") or []
    ram = float(profile.get("ram_gb") or 0)
    vram_total = max((g.get("vram_total_mb") or 0) for g in gpus) if gpus else 0

    if tier == HardwareTier.APPLE_UNIFIED:
        return max(0, int(ram * 1024) - _RAM_HEADROOM_RESERVE_MB)
    if tier == HardwareTier.CPU_ONLY:
        return max(0, int(ram * 1024) - _RAM_HEADROOM_RESERVE_MB)
    return int(vram_total or max(0, int(ram * 1024) - _RAM_HEADROOM_RESERVE_MB))


def _vram_headroom_mb(gpus: list[dict[str, Any]]) -> int:
    if not gpus:
        return 0
    best = 0
    for g in gpus:
        total = g.get("vram_total_mb") or 0
        used = g.get("vram_used_mb") or 0
        best = max(best, int(total - used))
    return best


_GPU_CAPACITY_RESERVE = 0.02
_RAM_HEADROOM_RESERVE_MB = 1024


def discrete_vram_total_mb(profile: dict[str, Any]) -> int:
    """Largest discrete GPU VRAM total from a hardware profile."""
    discrete = _discrete_gpu_entries(profile.get("gpus") or [])
    if not discrete:
        return 0
    return max((int(g.get("vram_total_mb") or 0) for g in discrete), default=0)


def fit_headroom_mb(profile: dict[str, Any]) -> int:
    """Budget for load blocking — GPU capacity on discrete cards, free memory elsewhere."""
    total = discrete_vram_total_mb(profile)
    if total > 0:
        return int(total * (1.0 - _GPU_CAPACITY_RESERVE))
    return vram_headroom_mb(profile)


def ram_headroom_mb(profile: dict[str, Any]) -> int:
    """Free system RAM for mmap and host-side llama.cpp allocations."""
    try:
        import psutil  # type: ignore

        avail = psutil.virtual_memory().available / (1024**2)
        return max(0, int(avail) - _RAM_HEADROOM_RESERVE_MB)
    except ImportError:
        ram = float(profile.get("ram_gb") or 8)
        return max(0, int(ram * 1024 * 0.45) - _RAM_HEADROOM_RESERVE_MB)


def vram_headroom_mb(profile: dict[str, Any]) -> int:
    """Free memory headroom for fit checks, using measured available memory."""
    gpus = profile.get("gpus") or []
    if gpus:
        best = _vram_headroom_mb(gpus)
        if best > 0:
            return best
    tier = classify_tier(profile)
    if tier in (HardwareTier.APPLE_UNIFIED, HardwareTier.CPU_ONLY):
        return ram_headroom_mb(profile)
    return effective_budget_mb(profile)


def memory_headroom_label(profile: dict[str, Any]) -> str:
    """Human label for free memory (RAM on Apple/CPU, VRAM on discrete GPU)."""
    tier = classify_tier(profile)
    if tier in (HardwareTier.APPLE_UNIFIED, HardwareTier.CPU_ONLY):
        return "RAM"
    return "VRAM"
