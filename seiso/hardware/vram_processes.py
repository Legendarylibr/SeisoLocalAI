"""Detect other GPU processes that can affect full model offload.

Advisory only: never blocks inference or model load.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from seiso.hardware.probes.common import GpuMemoryProcess
from seiso.hardware.probes.nvidia import (
    parse_nvidia_smi_process_csv,
    query_nvidia_compute_processes,
)

logger = logging.getLogger(__name__)

# Ignore tiny holders when listing processes (noise floor, not a model-size gate).
_MIN_VISIBLE_PROCESS_MB = 256

# Back-compat aliases for tests.
_parse_nvidia_smi_process_csv = parse_nvidia_smi_process_csv

__all__ = [
    "GpuMemoryProcess",
    "external_gpu_compute_processes",
    "log_vram_contention_at_startup",
    "query_gpu_compute_processes",
    "vram_contention_summary",
    "warn_before_large_model_load",
    "warn_before_model_load",
    "warn_vram_contention",
]


def query_gpu_compute_processes() -> list[GpuMemoryProcess]:
    """Return CUDA/compute processes currently using discrete GPU memory."""
    return query_nvidia_compute_processes()


def external_gpu_compute_processes(
    *, exclude_pid: int | None = None
) -> list[GpuMemoryProcess]:
    """GPU compute processes other than the current PID."""
    current = exclude_pid if exclude_pid is not None else os.getpid()
    return [proc for proc in query_gpu_compute_processes() if proc.pid != current]


def _safe_free_vram_mb() -> int:
    try:
        from seiso.memory.protection import headroom_mb

        return max(int(headroom_mb()), 0)
    except Exception:
        return 0


def _estimate_model_mb(model_path: str | None, est_mb: int = 0) -> int:
    if est_mb > 0:
        return int(est_mb)
    if not model_path:
        return 0
    try:
        from seiso.memory.protection import estimate_path_vram_mb

        return max(int(estimate_path_vram_mb(model_path)), 0)
    except Exception:
        return 0


def vram_contention_summary(
    *,
    exclude_pid: int | None = None,
    min_process_mb: int = _MIN_VISIBLE_PROCESS_MB,
    model_est_mb: int = 0,
) -> dict[str, Any]:
    """Summarize non-Seiso GPU memory use for API/logging."""
    processes = external_gpu_compute_processes(exclude_pid=exclude_pid)
    visible = [proc for proc in processes if proc.used_mb >= min_process_mb]
    total_mb = sum(proc.used_mb for proc in visible)
    free_mb = _safe_free_vram_mb()
    # Contended when this model may not fully fit and others hold VRAM.
    contended = total_mb > 0 and free_mb < model_est_mb if model_est_mb > 0 else total_mb > 0
    return {
        "external_vram_mb": total_mb,
        "free_vram_mb": free_mb,
        "model_est_mb": int(model_est_mb),
        "contended": contended,
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
    model_path: str | None = None,
    context: str = "model load",
) -> dict[str, Any] | None:
    """
    Advisory-only log when other GPU processes may limit offload for this model.

    Interprets the model being loaded (estimate from path when est is omitted).
    Never raises and never blocks inference or load.
    """
    try:
        est_mb = _estimate_model_mb(model_path, model_est_mb)
        if est_mb <= 0:
            # No model to interpret yet (startup / unknown path).
            return None

        summary = vram_contention_summary(model_est_mb=est_mb)
        external_mb = int(summary.get("external_vram_mb") or 0)
        if external_mb <= 0 or not summary.get("processes"):
            return None

        free_mb = int(summary.get("free_vram_mb") or 0)
        # Only surface when free headroom is short of this model's estimate.
        if free_mb >= est_mb:
            return None

        model_label = model_name or (Path(model_path).name if model_path else "model")
        model_label = f"{model_label} (~{est_mb / 1024:.1f} GB)"

        logger.warning(
            "GPU VRAM contention before %s: ~%.1f GB used by other processes while loading %s "
            "(free ~%.1f GB).\n%s\n"
            "Close those processes for fuller GPU offload if generation is slow. "
            "Load continues without waiting.",
            context,
            external_mb / 1024,
            model_label,
            free_mb / 1024,
            _format_process_lines(summary["processes"]),
        )
        return summary
    except Exception:
        logger.debug("vram contention check failed", exc_info=True)
        return None


def log_vram_contention_at_startup() -> dict[str, Any] | None:
    """Startup hook kept for compatibility — contention is judged when a model loads."""
    return None


def warn_before_model_load(
    *, model_path: str, est_mb: int | None = None
) -> dict[str, Any] | None:
    """Non-blocking pre-load advisory based on the model being opened."""
    try:
        return warn_vram_contention(
            model_est_mb=int(est_mb or 0),
            model_path=model_path,
            model_name=Path(model_path).name,
            context="model load",
        )
    except Exception:
        logger.debug("pre-load VRAM advisory failed", exc_info=True)
        return None


# Back-compat alias.
warn_before_large_model_load = warn_before_model_load
