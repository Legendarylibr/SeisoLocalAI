"""Strip identifying hardware details from artifacts and persisted metrics.

Runtime detection (tiers, VRAM budgets) may use local probes for UX, but
checkpoints, manifests, DB rows, and provenance files must not record GPU model
names, CPU brands, or other host-identifying strings.
"""

from __future__ import annotations

import platform
from typing import Any

# Fields allowed on persisted GPU snapshots (no model names or PCI ids).
_PERSIST_GPU_KEYS = frozenset(
    {
        "index",
        "vram_total_mb",
        "vram_used_mb",
        "utilization_pct",
        "temperature_c",
        "allocated_bytes",
        "reserved_bytes",
        "total_bytes",
    }
)


def _generic_accelerator_label(vendor: str | None = None) -> str:
    v = (vendor or "").lower()
    if v == "nvidia":
        return "discrete_gpu"
    if v == "amd":
        return "discrete_gpu_amd"
    if v == "mlx" or platform.system() == "Darwin":
        return "unified_gpu"
    return "cpu"


def sanitize_gpu_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a storage-safe GPU metric dict without device names."""
    out: dict[str, Any] = {}
    for key in _PERSIST_GPU_KEYS:
        if key in record and record[key] is not None:
            out[key] = record[key]
    if "index" not in out and "index" in record:
        out["index"] = record["index"]
    return out


def sanitize_gpu_stats(stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [sanitize_gpu_record(s) for s in stats]


def sanitize_system_metric_point(point: dict[str, Any]) -> dict[str, Any]:
    """Remove identifying fields from a live system metrics snapshot."""
    out = dict(point)
    gpus = out.get("gpus")
    if isinstance(gpus, list):
        out["gpus"] = [sanitize_gpu_record(g) if isinstance(g, dict) else g for g in gpus]
    return out


def sanitize_env_report(report: dict[str, Any]) -> dict[str, Any]:
    """Environment block for manifests — versions and capability flags only."""
    sanitized: dict[str, Any] = {
        "python_version": report.get("python", ""),
        "os_family": platform.system().lower(),
        "arch": platform.machine(),
        "collected_at": report.get("collected_at"),
        "git_commit": report.get("git_commit"),
        "torch_version": report.get("torch"),
        "cuda_available": bool(report.get("cuda_available")),
        "accelerator_class": _generic_accelerator_label(
            "nvidia" if report.get("cuda_available") else None
        ),
    }
    if report.get("pip_freeze"):
        sanitized["pip_freeze"] = report["pip_freeze"]
    return sanitized


def sanitize_hardware_profile_for_storage(profile: dict[str, Any]) -> dict[str, Any]:
    """Tier and budget fields only — no CPU/GPU product strings."""
    return {
        "tier": profile.get("tier"),
        "tier_label": profile.get("tier_label"),
        "backend": profile.get("backend"),
        "gpu_count": len(profile.get("gpus") or []),
        "ram_gb": profile.get("ram_gb"),
        "effective_vram_mb": profile.get("effective_vram_mb"),
        "vram_headroom_mb": profile.get("vram_headroom_mb"),
        "arch": profile.get("arch"),
        "local_only": True,
    }


def sanitize_label_for_display(raw: str, *, max_len: int = 64) -> str:
    """Sanitize hardware strings for ephemeral UI (not for persistence).

    Delegates to seiso.hardware.gpus.sanitize_hardware_label to avoid
    duplicated regex / logic.
    """
    from seiso.hardware.gpus import sanitize_hardware_label

    return sanitize_hardware_label(raw, max_len=max_len)
