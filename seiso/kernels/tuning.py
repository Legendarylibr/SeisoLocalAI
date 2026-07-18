"""CUDA kernel tuning profiles for RL-driven launch configuration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# Discrete profiles the RL policy selects.  ``auto`` matches the default heuristics
# in the CUDA sources; named profiles force specific launch paths.
KERNEL_PROFILES: tuple[dict[str, Any], ...] = (
    {"id": 0, "name": "auto", "rms_mode": 0, "swiglu_vec": 0, "lora_tile": 0},
    {"id": 1, "name": "stripe", "rms_mode": 1, "swiglu_vec": 8, "lora_tile": 128},
    # Name kept for RL/API compatibility; rms_mode 2 is a no-op alias of stripe.
    {"id": 2, "name": "parallax", "rms_mode": 2, "swiglu_vec": 8, "lora_tile": 512},
    {"id": 3, "name": "narrow_opt", "rms_mode": 1, "swiglu_vec": 4, "lora_tile": 128},
    {
        "id": 4,
        "name": "wide_throughput",
        "rms_mode": 1,
        "swiglu_vec": 8,
        "lora_tile": 512,
    },
    {"id": 5, "name": "balanced", "rms_mode": 0, "swiglu_vec": 8, "lora_tile": 256},
    {"id": 6, "name": "hopper_fa3", "rms_mode": 1, "swiglu_vec": 8, "lora_tile": 384},
    {"id": 7, "name": "blackwell", "rms_mode": 1, "swiglu_vec": 8, "lora_tile": 512},
)

_ACTIVE_PROFILE_ID = 0


@dataclass(frozen=True)
class KernelBenchmarkResult:
    profile_id: int
    profile_name: str
    latency_ms: float
    speedup_vs_pytorch: float
    memory_overhead_mb: float
    source: str


def kernel_profile_count() -> int:
    return len(KERNEL_PROFILES)


def kernel_profile_by_id(profile_id: int) -> dict[str, Any]:
    index = max(0, min(len(KERNEL_PROFILES) - 1, int(profile_id)))
    return KERNEL_PROFILES[index]


def active_kernel_profile_id() -> int:
    return _ACTIVE_PROFILE_ID


def apply_kernel_profile(profile_id: int) -> dict[str, Any]:
    """Set the process-wide kernel launch profile (CUDA extension when loaded)."""
    global _ACTIVE_PROFILE_ID
    profile = kernel_profile_by_id(profile_id)
    _ACTIVE_PROFILE_ID = int(profile["id"])

    arch_sm = 0
    use_graphs = 0
    use_overlap = 1
    try:
        from seiso.kernels.arch_tuning import detect_arch_tuning

        arch = detect_arch_tuning()
        arch_sm = arch.sm
        use_graphs = 1 if arch.use_cuda_graphs else 0
        use_overlap = 1 if arch.use_stream_overlap else 0
    except ImportError:
        pass

    try:
        from seiso.kernels.cuda_ops import set_kernel_tuning

        set_kernel_tuning(
            int(profile["rms_mode"]),
            int(profile["swiglu_vec"]),
            int(profile["lora_tile"]),
            arch_sm=arch_sm,
            use_cuda_graphs=use_graphs,
            use_stream_overlap=use_overlap,
        )
    except (ImportError, AttributeError, RuntimeError):
        pass

    return dict(profile)


def analytic_kernel_speedup(
    profile_id: int,
    *,
    hidden_dim: int,
    batch_rows: int,
    hardware_compute_factor: float = 1.0,
) -> float:
    """Fast analytic speedup estimate used by the simulator backend."""
    profile = kernel_profile_by_id(profile_id)
    wide = hidden_dim >= 4096
    rms_mode = int(profile["rms_mode"])
    swiglu_vec = int(profile["swiglu_vec"])

    speedup = 1.0
    if rms_mode in (1, 2):  # stripe (2 = legacy parallax alias)
        speedup *= 1.06 if not wide else 0.94
    elif rms_mode == 0:  # auto
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


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _bench_ms(fn, *, warmup: int = 3, iters: int = 12) -> float:
    for _ in range(warmup):
        fn()
    _sync_cuda()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync_cuda()
    return (time.perf_counter() - t0) / iters * 1000.0


@lru_cache(maxsize=256)
def _cached_live_benchmark(
    profile_id: int,
    hidden_dim: int,
    batch_rows: int,
    dtype: str,
) -> KernelBenchmarkResult:
    profile = kernel_profile_by_id(profile_id)
    try:
        import torch

        from seiso.kernels.dispatch import fused_rms_norm, fused_swiglu
    except ImportError:
        speedup = analytic_kernel_speedup(profile_id, hidden_dim=hidden_dim, batch_rows=batch_rows)
        return KernelBenchmarkResult(
            profile_id=profile_id,
            profile_name=str(profile["name"]),
            latency_ms=1.0 / max(speedup, 0.1),
            speedup_vs_pytorch=speedup,
            memory_overhead_mb=0.0,
            source="analytic_fallback",
        )

    if not torch.cuda.is_available():
        speedup = analytic_kernel_speedup(profile_id, hidden_dim=hidden_dim, batch_rows=batch_rows)
        return KernelBenchmarkResult(
            profile_id=profile_id,
            profile_name=str(profile["name"]),
            latency_ms=1.0,
            speedup_vs_pytorch=speedup,
            memory_overhead_mb=0.0,
            source="analytic_no_cuda",
        )

    try:
        from seiso.kernels.cuda_env import configure_cuda_build_env

        configure_cuda_build_env()
        apply_kernel_profile(profile_id)
        device = "cuda"
        torch_dtype = getattr(torch, dtype, torch.bfloat16)
        rows = max(64, int(batch_rows))
        # Vectorized SwiGLU kernels require hidden dim aligned to 8.
        cols = max(128, int(hidden_dim))
        cols = ((cols + 7) // 8) * 8

        x = torch.randn(rows, cols, device=device, dtype=torch_dtype)
        w = torch.ones(cols, device=device, dtype=torch_dtype)
        gate = torch.randn(rows, cols, device=device, dtype=torch_dtype)
        up = torch.randn(rows, cols, device=device, dtype=torch_dtype)

        def pytorch_rms():
            y = x
            v = y.pow(2).mean(dim=-1, keepdim=True)
            return y * torch.rsqrt(v + 1e-6) * w

        def fused_rms():
            return fused_rms_norm(x, w)

        def pytorch_swiglu():
            return torch.nn.functional.silu(gate) * up

        def fused_sw():
            return fused_swiglu(gate, up)

        pt_ms = _bench_ms(pytorch_rms) + _bench_ms(pytorch_swiglu)
        fused_ms = _bench_ms(fused_rms) + _bench_ms(fused_sw)
        speedup = pt_ms / max(fused_ms, 1e-6)

        return KernelBenchmarkResult(
            profile_id=profile_id,
            profile_name=str(profile["name"]),
            latency_ms=float(fused_ms),
            speedup_vs_pytorch=float(speedup),
            memory_overhead_mb=0.0,
            source="live_cuda",
        )
    except Exception as exc:
        logger.warning(
            "Live CUDA kernel benchmark failed for profile %d: %s",
            profile_id,
            exc,
        )
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        speedup = analytic_kernel_speedup(profile_id, hidden_dim=hidden_dim, batch_rows=batch_rows)
        return KernelBenchmarkResult(
            profile_id=profile_id,
            profile_name=str(profile["name"]),
            latency_ms=1.0 / max(speedup, 0.1),
            speedup_vs_pytorch=speedup,
            memory_overhead_mb=0.0,
            source="analytic_benchmark_failed",
        )


def benchmark_kernel_profile(
    profile_id: int,
    *,
    hidden_dim: int = 4096,
    batch_rows: int = 4096,
    dtype: str = "bfloat16",
    live: bool = True,
) -> KernelBenchmarkResult:
    """Benchmark a kernel profile; uses LRU cache for training throughput."""
    if not live:
        speedup = analytic_kernel_speedup(profile_id, hidden_dim=hidden_dim, batch_rows=batch_rows)
        profile = kernel_profile_by_id(profile_id)
        return KernelBenchmarkResult(
            profile_id=profile_id,
            profile_name=str(profile["name"]),
            latency_ms=1.0 / max(speedup, 0.1),
            speedup_vs_pytorch=speedup,
            memory_overhead_mb=0.0,
            source="analytic",
        )
    return _cached_live_benchmark(profile_id, hidden_dim, batch_rows, dtype)


def kernel_metrics_dict(
    profile_id: int,
    *,
    hidden_dim: int,
    batch_rows: int,
    live_benchmark: bool = False,
    hardware_compute_factor: float = 1.0,
) -> dict[str, float | str]:
    """Metrics merged into RL backend evaluation for reward computation."""
    bench = benchmark_kernel_profile(
        profile_id,
        hidden_dim=hidden_dim,
        batch_rows=batch_rows,
        live=live_benchmark,
    )
    speedup = bench.speedup_vs_pytorch
    if bench.source.startswith("analytic"):
        speedup = analytic_kernel_speedup(
            profile_id,
            hidden_dim=hidden_dim,
            batch_rows=batch_rows,
            hardware_compute_factor=hardware_compute_factor,
        )
    profile = kernel_profile_by_id(profile_id)
    return {
        "kernel_profile_id": float(profile_id),
        "kernel_profile_name": str(profile["name"]),
        "kernel_latency_ms": float(bench.latency_ms),
        "kernel_speedup": float(speedup),
        "kernel_memory_overhead_mb": float(bench.memory_overhead_mb),
        "kernel_benchmark_source": str(bench.source),
    }


__all__ = [
    "KERNEL_PROFILES",
    "KernelBenchmarkResult",
    "active_kernel_profile_id",
    "analytic_kernel_speedup",
    "apply_kernel_profile",
    "benchmark_kernel_profile",
    "kernel_metrics_dict",
    "kernel_profile_by_id",
    "kernel_profile_count",
]
