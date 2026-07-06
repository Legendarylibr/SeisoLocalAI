"""Load preflight, headroom probes, and HF max_memory."""

from __future__ import annotations

import logging
import platform
from pathlib import Path
from typing import Any

from seiso import platform as seiso_platform
from seiso.env import env_bool
from seiso.hardware import assess_hardware_fit, hardware_profile, vram_headroom_mb
from seiso.hardware.tiers import fit_headroom_mb
from seiso.memory.protection._facade import protection
from seiso.memory.protection.constants import _DEFAULT_RESERVE_RATIO
from seiso.memory.protection.oom import MemoryLoadBlockedError, allow_memory_overcommit

logger = logging.getLogger(__name__)


def headroom_mb() -> int:
    """Free memory headroom in MB for fit labels and status reporting."""
    profile = protection().hardware_profile()
    try:
        return int(vram_headroom_mb(profile))
    except Exception:
        gpus = profile.get("gpus") or []
        if gpus:
            best = 0
            for gpu in gpus:
                total = int(gpu.get("vram_total_mb") or 0)
                used = int(gpu.get("vram_used_mb") or 0)
                if total > 0:
                    best = max(best, max(total - used, 0))
            if best > 0:
                return best
        ram = float(profile.get("ram_gb") or 8)
        avail = available_ram_mb()
        if avail > 0:
            return int(min(avail * 0.72, ram * 1024 * 0.45))
        return int(ram * 1024 * 0.35)


def available_ram_mb() -> int:
    """Cross-platform available RAM in MB (Linux, macOS, Windows)."""
    try:
        import psutil

        return int(psutil.virtual_memory().available / (1024**2))
    except Exception:
        pass
    if platform.system() == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            windll = getattr(ctypes, "windll", None)
            if windll is not None and windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys / (1024**2))
        except Exception:
            pass
    return int(float(protection().hardware_profile().get("ram_gb") or 8) * 1024 * 0.5)


def build_hf_max_memory(*, reserve_ratio: float = _DEFAULT_RESERVE_RATIO) -> dict[int, str] | None:
    """Build HuggingFace ``max_memory`` unless caps are explicitly disabled."""
    if env_bool("SEISO_DISABLE_MEMORY_CAPS", False):
        return None
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    max_memory: dict[int, str] = {}
    for i in range(torch.cuda.device_count()):
        try:
            free_bytes, _total = torch.cuda.mem_get_info(i)
        except Exception:
            props = torch.cuda.get_device_properties(i)
            free_bytes = int(props.total_memory * (1.0 - reserve_ratio))
        usable = max(int(free_bytes * (1.0 - reserve_ratio)), 256 * 1024**2)
        max_memory[i] = f"{usable // (1024**2)}MiB"
    return max_memory or None


def assess_path_memory_fit(path: str | Path, *, mode: str = "chat") -> dict[str, Any]:
    """Return fit metadata compatible with Forge hardware assessments."""
    from seiso.memory.protection._facade import protection

    p = Path(path).expanduser()
    est_mb = protection().estimate_path_vram_mb(p, mode=mode)
    if p.is_file() and p.suffix.lower() == ".gguf":
        try:
            from seiso.inference.llama_vision import resolve_mmproj_path

            mmproj = resolve_mmproj_path(p)
            if mmproj:
                est_mb += protection().estimate_path_vram_mb(mmproj, mode=mode)
        except ImportError:
            pass
    est_gb = round(est_mb / 1024, 2)
    profile = protection().hardware_profile()
    try:
        return assess_hardware_fit(est_gb, profile, mode=mode)
    except Exception:
        capacity = int(fit_headroom_mb(profile))
        free = int(vram_headroom_mb(profile))
        raw_budget = free if free > 0 else capacity
        reserve = max(256, int(raw_budget * 0.02)) if raw_budget > 0 else 0
        budget = max(0, raw_budget - reserve)
        budget_exceeded = budget > 0 and est_mb > budget
        blocked = mode != "chat" and budget_exceeded
        budget_gb = round(budget / 1024, 1)
        return {
            "hardware_fit": "unlikely" if budget_exceeded else "good",
            "est_vram_mb": est_mb,
            "memory_load_blocked": blocked,
            "memory_load_budget_exceeded": budget_exceeded,
            "memory_load_blocked_reason": (
                f"Needs ~{est_gb:.1f} GB at runtime but only ~{budget_gb} GB is safely available right now."
                if blocked
                else None
            ),
        }


_LLAMACPP_DEFER_WARNINGS: dict[str, str] = {
    "apple_unified": (
        "Low free unified memory — trying llama.cpp with mmap plus Mac CPU "
        "offload fallback. Close apps if loading still fails."
    ),
    "linux_nvidia": (
        "Low free VRAM — trying full GPU offload with conservative batch limits. "
        "Close other GPU apps if loading still fails."
    ),
}


def assess_path_memory_fit_for_load(
    path: str | Path,
    *,
    mode: str = "chat",
    pool: Any | None = None,
    backend: str | None = None,
    unload_if_needed: bool = True,
) -> dict[str, Any]:
    """Assess fit after unloading any active Seiso model that would be replaced."""
    from seiso.inference.model_pool import get_model_pool
    from seiso.memory.protection._facade import protection

    active_pool = pool or get_model_pool()
    if unload_if_needed:
        active_pool.prepare_for_load(str(path), backend)
    fit = protection().assess_path_memory_fit(path, mode=mode)
    profile = protection().hardware_profile()
    defer = _llamacpp_deferred_preflight_platform(fit, backend=backend, mode=mode, profile=profile)
    if defer:
        fit = dict(fit)
        fit["memory_load_blocked"] = False
        fit["memory_load_blocked_reason"] = None
        fit["memory_load_warning"] = _LLAMACPP_DEFER_WARNINGS.get(
            defer,
            "Low free memory — trying llama.cpp with conservative fallbacks.",
        )
    return fit


def _llamacpp_deferred_preflight_platform(
    fit: dict[str, Any],
    *,
    backend: str | None,
    mode: str,
    profile: dict[str, Any] | None = None,
) -> str | None:
    """Return platform id when llama.cpp should try load despite preflight block."""
    if mode != "chat":
        return None
    if str(backend or "").lower() not in {"llamacpp", "llama"}:
        return None

    blocked = bool(fit.get("memory_load_blocked"))
    low_free = bool(fit.get("memory_load_budget_exceeded")) and not blocked
    if not blocked and not low_free:
        return None

    defer = seiso_platform.llamacpp_deferred_preflight_platform(profile=profile)
    if not defer:
        return None

    if defer == "linux_nvidia":
        est_mb = int(fit.get("est_vram_mb") or 0)
        try:
            capacity_mb = fit_headroom_mb(profile or hardware_profile())
        except Exception:
            return None
        if est_mb <= 0 or capacity_mb <= 0 or est_mb > capacity_mb:
            return None
        try:
            from seiso.inference.model_pool import _llama_gpu_offload_ok

            if not _llama_gpu_offload_ok():
                return None
        except ImportError:
            pass
    return defer


def ensure_load_fits(
    path: str | Path,
    *,
    mode: str = "chat",
    backend: str | None = None,
) -> dict[str, Any]:
    """Apply load preflight policy while leaving chat loads best-effort."""
    from seiso.memory.protection._facade import protection

    fit = protection().assess_path_memory_fit_for_load(path, mode=mode, backend=backend)
    backend_key = str(backend or "").lower()
    llamacpp_backend = backend_key in {"llamacpp", "llama"}
    if fit.get("memory_load_blocked"):
        reason = fit.get("memory_load_blocked_reason") or "Model exceeds available memory"
        if mode == "chat":
            fit = dict(fit)
            fit["memory_load_blocked"] = False
            fit["memory_load_blocked_reason"] = None
            fit["memory_load_warning"] = reason
            logger.warning("Inference memory preflight advisory: %s", reason)
        elif allow_memory_overcommit():
            logger.warning("Memory overcommit allowed: %s", reason)
        else:
            raise MemoryLoadBlockedError(reason)
    if (
        mode == "chat"
        and fit.get("memory_load_budget_exceeded")
        and not llamacpp_backend
    ):
        est_gb = round(int(fit.get("est_vram_mb") or 0) / 1024, 1)
        reason = (
            f"Needs ~{est_gb:.1f} GB at runtime but free memory is low right now. "
            "Free memory or use llama.cpp for tiered GPU load fallbacks."
        )
        logger.warning("Inference memory preflight advisory: %s", reason)
    return fit

