"""Detect other GPU processes that can block full model offload."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Compute workloads above this are worth surfacing before large GGUF loads.
_DEFAULT_WARN_EXTERNAL_MB = 2048
_DEFAULT_STARTUP_WARN_MB = 4096
_LARGE_MODEL_MB = 6000


@dataclass(frozen=True)
class GpuMemoryProcess:
    pid: int
    process_name: str
    used_mb: int


def _parse_nvidia_smi_process_csv(stdout: str) -> list[GpuMemoryProcess]:
    processes: list[GpuMemoryProcess] = []
    for line in stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        name = parts[1] or "unknown"
        used_raw = parts[2].replace(" MiB", "").strip()
        if used_raw in ("", "[N/A]", "N/A"):
            continue
        try:
            used_mb = int(float(used_raw))
        except ValueError:
            continue
        if used_mb <= 0:
            continue
        processes.append(GpuMemoryProcess(pid=pid, process_name=name, used_mb=used_mb))
    return processes


def query_gpu_compute_processes() -> list[GpuMemoryProcess]:
    """Return CUDA/compute processes currently using discrete GPU memory."""
    try:
        from seiso.security.nvidia_boundary import (
            _run_nvidia_smi,
            resolve_nvidia_smi_executable,
        )
    except ImportError:
        return []

    exe = resolve_nvidia_smi_executable()
    if not exe:
        return []

    for fields in (
        "pid,process_name,used_gpu_memory",
        "pid,process_name,used_memory",
    ):
        proc = _run_nvidia_smi(
            exe,
            f"--query-compute-apps={fields}",
            "--format=csv,noheader,nounits",
            timeout=3,
        )
        if proc is None or proc.returncode != 0 or not proc.stdout.strip():
            continue
        parsed = _parse_nvidia_smi_process_csv(proc.stdout)
        if parsed:
            return parsed
    return []


def external_gpu_compute_processes(
    *, exclude_pid: int | None = None
) -> list[GpuMemoryProcess]:
    """GPU compute processes other than the current PID."""
    current = exclude_pid if exclude_pid is not None else os.getpid()
    return [proc for proc in query_gpu_compute_processes() if proc.pid != current]


def vram_contention_summary(
    *,
    exclude_pid: int | None = None,
    min_process_mb: int = 256,
    warn_external_mb: int = _DEFAULT_WARN_EXTERNAL_MB,
) -> dict[str, Any]:
    """Summarize non-Seiso GPU memory use for API/logging."""
    processes = external_gpu_compute_processes(exclude_pid=exclude_pid)
    visible = [proc for proc in processes if proc.used_mb >= min_process_mb]
    total_mb = sum(proc.used_mb for proc in visible)
    return {
        "external_vram_mb": total_mb,
        "contended": total_mb >= warn_external_mb,
        "processes": [
            {
                "pid": proc.pid,
                "name": proc.process_name,
                "used_mb": proc.used_mb,
            }
            for proc in sorted(visible, key=lambda item: item.used_mb, reverse=True)
        ],
    }


def _format_process_lines(processes: list[dict[str, Any]], *, limit: int = 4) -> str:
    lines: list[str] = []
    for proc in processes[:limit]:
        used_gb = proc["used_mb"] / 1024
        lines.append(f"  - {proc['name']} (PID {proc['pid']}): {used_gb:.1f} GB")
    extra = len(processes) - limit
    if extra > 0:
        lines.append(f"  - …and {extra} more (run `nvidia-smi` for details)")
    return "\n".join(lines)


def warn_vram_contention(
    *,
    model_est_mb: int = 0,
    model_name: str | None = None,
    context: str = "model load",
    warn_external_mb: int = _DEFAULT_WARN_EXTERNAL_MB,
    large_model_mb: int = _LARGE_MODEL_MB,
) -> dict[str, Any] | None:
    """
    Log a warning when other GPU processes may force partial offload.

    Returns the contention summary when a warning is emitted.
    """
    summary = vram_contention_summary(warn_external_mb=warn_external_mb)
    if not summary["contended"]:
        return None

    if model_est_mb >= large_model_mb:
        should_warn = True
    elif model_est_mb > 0:
        should_warn = summary["external_vram_mb"] >= max(
            warn_external_mb, model_est_mb // 4
        )
    else:
        should_warn = summary["external_vram_mb"] >= _DEFAULT_STARTUP_WARN_MB

    if not should_warn:
        return None

    model_label = model_name or "large model"
    if model_est_mb > 0:
        model_label = f"{model_label} (~{model_est_mb / 1024:.1f} GB)"

    logger.warning(
        "GPU VRAM contention before %s: ~%.1f GB used by other processes while loading %s.\n%s\n"
        "Close those processes for full GPU offload and faster generation.",
        context,
        summary["external_vram_mb"] / 1024,
        model_label,
        _format_process_lines(summary["processes"]),
    )
    return summary


def log_vram_contention_at_startup() -> dict[str, Any] | None:
    """Forge/runtime startup hook — warn when GPU is already heavily occupied."""
    return warn_vram_contention(context="startup")


def warn_before_large_model_load(
    *, model_path: str, est_mb: int
) -> dict[str, Any] | None:
    """Pre-load hook for GGUF models that benefit from a clean GPU."""
    from pathlib import Path

    return warn_vram_contention(
        model_est_mb=est_mb,
        model_name=Path(model_path).name,
        context="model load",
    )
