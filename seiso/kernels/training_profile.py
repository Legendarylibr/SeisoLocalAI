"""CUDA training profile — auto-tune kernels for speed vs VRAM on NVIDIA."""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any

from seiso.compat import StrEnum

logger = logging.getLogger(__name__)

_HEADROOM_LEAN_MB = 8192
_HEADROOM_SPEED_MB = 16384


class CudaTrainingMode(StrEnum):
    SPEED = "speed"
    BALANCED = "balanced"
    LEAN = "lean"


_PROFILE_BY_MODE: dict[CudaTrainingMode, int] = {
    CudaTrainingMode.SPEED: 4,  # wide_throughput
    CudaTrainingMode.BALANCED: 5,  # balanced
    CudaTrainingMode.LEAN: 3,  # narrow_opt
}

_LAST_PROFILE: dict[str, Any] | None = None


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def guess_hidden_dim(model_id: str) -> int:
    """Rough hidden size from model id for kernel micro-benchmarks."""
    lowered = model_id.lower()
    if re.search(r"\b(70b|72b|65b|34b|32b)\b", lowered):
        return 8192
    if re.search(r"\b(13b|14b|12b)\b", lowered):
        return 5120
    if re.search(r"\b(1b|3b)\b", lowered):
        return 2048
    return 4096


def resolve_cuda_training_mode(*, headroom_mb: int, est_train_mb: int = 0) -> CudaTrainingMode:
    if headroom_mb > 0 and headroom_mb < _HEADROOM_LEAN_MB:
        return CudaTrainingMode.LEAN
    if headroom_mb >= _HEADROOM_SPEED_MB:
        if est_train_mb <= 0 or est_train_mb < int(headroom_mb * 0.6):
            return CudaTrainingMode.SPEED
    return CudaTrainingMode.BALANCED


def native_cuda_kernels_available() -> bool:
    try:
        from seiso.kernels.cuda_ops import is_cuda_available
        from seiso.kernels.platform import GpuVendor, detect_gpu

        return detect_gpu().vendor == GpuVendor.NVIDIA and is_cuda_available()
    except ImportError:
        return False


@lru_cache(maxsize=8)
def auto_select_kernel_profile(hidden_dim: int, batch_rows: int) -> int:
    """Micro-benchmark CUDA kernel profiles; cached per (hidden, batch_rows)."""
    if not native_cuda_kernels_available():
        return _PROFILE_BY_MODE[CudaTrainingMode.BALANCED]

    from seiso.kernels.tuning import apply_kernel_profile, benchmark_kernel_profile

    candidates = (4, 2, 5, 1, 3)  # throughput-first search order
    best_id = _PROFILE_BY_MODE[CudaTrainingMode.BALANCED]
    best_ms = float("inf")
    for profile_id in candidates:
        bench = benchmark_kernel_profile(
            profile_id,
            hidden_dim=hidden_dim,
            batch_rows=batch_rows,
            live=True,
        )
        if bench.latency_ms < best_ms:
            best_ms = bench.latency_ms
            best_id = profile_id

    apply_kernel_profile(best_id)
    logger.info(
        "CUDA kernel auto-tune: profile=%d latency=%.3fms (hidden=%d rows=%d)",
        best_id,
        best_ms,
        hidden_dim,
        batch_rows,
    )
    return best_id


def apply_cuda_speedopts(*, deterministic: bool) -> None:
    """Enable TF32 / cuDNN autotune when reproducibility is not required."""
    if deterministic:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except (ImportError, AttributeError):
        pass


def prepare_cuda_training_profile(
    *,
    headroom_mb: int,
    est_train_mb: int = 0,
    model_id: str = "",
    batch_size: int = 1,
    max_seq_length: int = 2048,
) -> dict[str, Any]:
    """
    Select CUDA kernel tuning + training flags for best speed at minimal VRAM.

    Returns a dict merged into TrainConfig by ``apply_training_memory_guards``.
    """
    mode = resolve_cuda_training_mode(headroom_mb=headroom_mb, est_train_mb=est_train_mb)
    low_vram = mode == CudaTrainingMode.LEAN
    hidden_dim = guess_hidden_dim(model_id)
    batch_rows = max(64, int(batch_size) * int(max_seq_length))

    if low_vram:
        os.environ.setdefault("SEISO_KERNEL_LOW_VRAM", "1")
    elif os.environ.get("SEISO_KERNEL_LOW_VRAM", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        os.environ.pop("SEISO_KERNEL_LOW_VRAM", None)

    profile_id = _PROFILE_BY_MODE[mode]
    if _env_flag("SEISO_KERNEL_AUTO_TUNE", default=native_cuda_kernels_available()):
        profile_id = auto_select_kernel_profile(hidden_dim, batch_rows)
    else:
        from seiso.kernels.tuning import apply_kernel_profile

        apply_kernel_profile(profile_id)

    if low_vram:
        from seiso.kernels.memory_mode import apply_low_vram_kernel_tuning

        apply_low_vram_kernel_tuning()

    if mode == CudaTrainingMode.SPEED:
        gradient_checkpointing = False
    elif mode == CudaTrainingMode.LEAN:
        gradient_checkpointing = True
    else:
        gradient_checkpointing = est_train_mb > int(headroom_mb * 0.65) if headroom_mb > 0 else True

    caps_fused = True
    try:
        from seiso.training.platform_caps import training_capabilities

        caps = training_capabilities()
        caps_fused = bool(caps.get("fused_kernels_available"))
    except ImportError:
        pass

    result: dict[str, Any] = {
        "cuda_training_mode": mode.value,
        "kernel_profile_id": profile_id,
        "kernel_low_vram": low_vram,
        "gradient_checkpointing": gradient_checkpointing,
        "use_fused_ce": caps_fused,
        "use_triton": caps_fused,
        "use_fused_lora": caps_fused and native_cuda_kernels_available(),
    }

    if mode == CudaTrainingMode.LEAN and max_seq_length > 1024 and headroom_mb < 6144:
        result["max_seq_length"] = min(max_seq_length, 1024)

    logger.info(
        "CUDA training profile: mode=%s profile=%d low_vram=%s gc=%s fused=%s",
        mode.value,
        profile_id,
        low_vram,
        gradient_checkpointing,
        caps_fused,
    )
    global _LAST_PROFILE
    _LAST_PROFILE = result
    return result


def last_cuda_training_profile() -> dict[str, Any]:
    return dict(_LAST_PROFILE or {})
