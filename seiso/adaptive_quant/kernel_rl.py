"""Kernel RL helpers — profiles, analytic metrics, and decision finalization."""

from __future__ import annotations

from typing import Any

# Keep profile table in sync with seiso.kernels.tuning.KERNEL_PROFILES.
KERNEL_PROFILES: tuple[dict[str, Any], ...] = (
    {"id": 0, "name": "auto", "rms_mode": 0, "swiglu_vec": 0, "lora_tile": 0},
    {"id": 1, "name": "stripe", "rms_mode": 1, "swiglu_vec": 8, "lora_tile": 32},
    {"id": 2, "name": "parallax", "rms_mode": 2, "swiglu_vec": 8, "lora_tile": 64},
    {"id": 3, "name": "narrow_opt", "rms_mode": 1, "swiglu_vec": 4, "lora_tile": 16},
    {
        "id": 4,
        "name": "wide_throughput",
        "rms_mode": 2,
        "swiglu_vec": 8,
        "lora_tile": 64,
    },
    {"id": 5, "name": "balanced", "rms_mode": 0, "swiglu_vec": 8, "lora_tile": 32},
)


def kernel_profile_count(config: Any | None = None) -> int:
    if config is not None and getattr(config, "kernel_profile_count", 0) > 0:
        return int(config.kernel_profile_count)
    return len(KERNEL_PROFILES)


def kernel_profile_by_id(profile_id: int, config: Any | None = None) -> dict[str, Any]:
    count = kernel_profile_count(config)
    index = max(0, min(count - 1, int(profile_id)))
    return KERNEL_PROFILES[index]


def analytic_kernel_speedup(
    profile_id: int,
    *,
    hidden_dim: int,
    batch_rows: int,
    hardware_compute_factor: float = 1.0,
    config: Any | None = None,
) -> float:
    profile = kernel_profile_by_id(profile_id, config)
    wide = hidden_dim >= 4096
    rms_mode = int(profile["rms_mode"])
    swiglu_vec = int(profile["swiglu_vec"])

    speedup = 1.0
    if rms_mode == 1:
        speedup *= 1.06 if not wide else 0.94
    elif rms_mode == 2:
        speedup *= 0.95 if not wide else 1.14
    elif rms_mode == 0:
        speedup *= 1.08 if wide else 1.03

    if swiglu_vec == 8:
        speedup *= 1.05 if hidden_dim >= 2048 else 1.02
    elif swiglu_vec == 4:
        speedup *= 1.01 if hidden_dim < 2048 else 0.97

    lora_tile = int(profile["lora_tile"])
    if lora_tile == 64:
        speedup *= 1.04
    elif lora_tile == 16:
        speedup *= 0.98

    batch_factor = min(1.12, 1.0 + (batch_rows / 8192.0) * 0.08)
    speedup *= batch_factor * max(0.85, min(1.25, hardware_compute_factor))
    return max(0.75, min(1.45, speedup))


def kernel_metrics_for_profile(
    profile_id: int,
    *,
    hidden_dim: int,
    batch_rows: int,
    hardware_compute_factor: float = 1.0,
    config: Any | None = None,
) -> dict[str, float | str]:
    speedup = analytic_kernel_speedup(
        profile_id,
        hidden_dim=hidden_dim,
        batch_rows=batch_rows,
        hardware_compute_factor=hardware_compute_factor,
        config=config,
    )
    profile = kernel_profile_by_id(profile_id, config)
    latency_ms = 1.0 / max(speedup, 0.1)
    return {
        "kernel_profile_id": float(profile_id),
        "kernel_profile_name": str(profile["name"]),
        "kernel_latency_ms": float(latency_ms),
        "kernel_speedup": float(speedup),
        "kernel_memory_overhead_mb": 0.0,
        "kernel_benchmark_source": "analytic",
    }


def finalize_kernel_profile(decision: Any, config: Any) -> None:
    """Normalize kernel profile selection stored on a quantization decision."""
    if not getattr(config, "kernel_rl_enabled", False):
        return
    metadata = dict(decision.metadata)
    raw_index = metadata.get("kernel_profile_index")
    if raw_index is None:
        raw_index = getattr(config, "kernel_default_profile", 0)
    count = kernel_profile_count(config)
    index = max(0, min(count - 1, int(raw_index)))
    profile = kernel_profile_by_id(index, config)
    metadata["kernel_profile_index"] = index
    metadata["kernel_profile_name"] = str(profile["name"])
    metadata["kernel_rms_mode"] = int(profile["rms_mode"])
    metadata["kernel_swiglu_vec"] = int(profile["swiglu_vec"])
    metadata["kernel_lora_tile"] = int(profile["lora_tile"])
    decision.metadata = metadata


def kernel_feedback_scalar(decision: Any, config: Any) -> float:
    if not getattr(config, "kernel_rl_enabled", False):
        return 0.0
    count = max(1, kernel_profile_count(config) - 1)
    index = int(decision.metadata.get("kernel_profile_index", 0))
    return float(index) / float(count)


__all__ = [
    "KERNEL_PROFILES",
    "analytic_kernel_speedup",
    "finalize_kernel_profile",
    "kernel_feedback_scalar",
    "kernel_metrics_for_profile",
    "kernel_profile_by_id",
    "kernel_profile_count",
]
