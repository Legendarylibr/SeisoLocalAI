"""NVIDIA GPU/process probing via nvidia-smi."""

from __future__ import annotations

import platform
import subprocess
from typing import Any

from seiso.hardware.probes.common import GpuMemoryProcess, sanitize_hardware_label


def probe_nvidia_gpus() -> list[dict[str, Any]]:
    """Enumerate GPUs via nvidia-smi when PyTorch CUDA is unavailable."""
    if platform.system().lower() not in {"linux", "windows"}:
        return []
    from seiso.security.nvidia_boundary import query_nvidia_gpus

    gpus: list[dict[str, Any]] = []
    for item in query_nvidia_gpus():
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        total_raw = item.get("memory_total_mb")
        total_mb = (
            int(total_raw)
            if isinstance(total_raw, (int, float)) and total_raw > 0
            else None
        )
        gpus.append(
            {
                "index": int(item.get("index", len(gpus))),
                "name": sanitize_hardware_label(name),
                "vram_total_mb": total_mb,
                "vram_used_mb": None,
                "utilization_pct": None,
                "temperature_c": None,
            }
        )
    return gpus


def nvidia_gpu_metrics() -> dict[int, dict[str, float]]:
    """Live utilization/VRAM/temperature per GPU index from nvidia-smi."""
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


def parse_nvidia_smi_process_csv(stdout: str) -> list[GpuMemoryProcess]:
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


def query_nvidia_compute_processes() -> list[GpuMemoryProcess]:
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
        parsed = parse_nvidia_smi_process_csv(proc.stdout)
        if parsed:
            return parsed
    return []
