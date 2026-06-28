"""Bridge Seiso CUDA kernel tuning into seiso.adaptive_quant RL evaluation."""

from __future__ import annotations

from typing import Any


def evaluate_kernel_for_decision(
    decision: Any,
    state: Any,
    config: Any,
) -> dict[str, float | str]:
    """Apply kernel profile and return metrics for reward computation."""
    from seiso.adaptive_quant.kernel_rl import kernel_metrics_for_profile

    profile_id = int(decision.metadata.get("kernel_profile_index", 0))
    hidden_dim = int(getattr(config, "kernel_hidden_dim", 4096))
    batch_rows = max(
        64,
        int(getattr(config, "kernel_batch_rows", 4096))
        * max(1, int(state.input_features.prompt_length) // 128),
    )
    live = bool(getattr(config, "kernel_rl_live_benchmark", False))

    if live:
        try:
            from seiso.kernels.tuning import apply_kernel_profile, kernel_metrics_dict

            apply_kernel_profile(profile_id)
            return kernel_metrics_dict(
                profile_id,
                hidden_dim=hidden_dim,
                batch_rows=batch_rows,
                live_benchmark=True,
                hardware_compute_factor=float(state.hardware_profile.compute_factor),
            )
        except ImportError:
            pass

    try:
        from seiso.kernels.tuning import apply_kernel_profile

        apply_kernel_profile(profile_id)
    except ImportError:
        pass

    return kernel_metrics_for_profile(
        profile_id,
        hidden_dim=hidden_dim,
        batch_rows=batch_rows,
        hardware_compute_factor=float(state.hardware_profile.compute_factor),
        config=config,
    )


def merge_kernel_metrics(
    metrics: dict[str, Any],
    kernel_metrics: dict[str, float | str],
    *,
    config: Any,
) -> dict[str, Any]:
    """Fold kernel metrics into backend metrics and adjust latency/throughput."""
    if not kernel_metrics:
        return metrics

    merged = dict(metrics)
    speedup = float(kernel_metrics.get("kernel_speedup", 1.0))
    merged.update(kernel_metrics)

    if speedup > 0.0:
        merged["latency_ms"] = float(merged["latency_ms"]) / speedup
        merged["throughput_tps"] = float(merged["throughput_tps"]) * speedup
        if "latency_ms_per_token" in merged:
            merged["latency_ms_per_token"] = float(merged["latency_ms_per_token"]) / speedup

    overhead = float(kernel_metrics.get("kernel_memory_overhead_mb", 0.0))
    if overhead > 0.0:
        merged["memory_mb"] = float(merged["memory_mb"]) + overhead

    merged["kernel_reward_source"] = str(kernel_metrics.get("kernel_benchmark_source", "analytic"))
    return merged


__all__ = ["evaluate_kernel_for_decision", "merge_kernel_metrics"]
