"""KV cache sizing and offload headroom math."""

from __future__ import annotations

from pathlib import Path

from seiso.env import env_bool
from seiso.inference.backends import gguf_total_layers
from seiso.memory.estimates import guess_params_from_name
from seiso.memory.protection._facade import protection
from seiso.memory.protection.constants import (
    _MAX_LLAMA_CTX,
    _MIN_LLAMA_BATCH,
    _MIN_LLAMA_CTX,
)


def _host_os_reserve_mb(ram_mb: int) -> int:
    return max(512, int(ram_mb * 0.08))


def _gpu_layer_fraction(n_gpu_layers: int, total_layers: int) -> float:
    if n_gpu_layers == -1:
        return 1.0
    return max(0.0, min(float(n_gpu_layers) / float(total_layers or 64), 1.0))


def llama_batch_headroom_mb(
    free_mb: int,
    *,
    model_path: str | Path | None = None,
    n_gpu_layers: int = -1,
    n_ctx: int = 2048,
    weights_resident: bool = False,
) -> int:
    """VRAM left for llama.cpp batch/KV after estimated weight offload."""
    if not model_path or n_gpu_layers == 0:
        return free_mb
    path = Path(model_path)
    try:
        weight_mb = int(protection().estimate_path_vram_mb(path))
        total_layers = gguf_total_layers(path) if path.is_file() else 64
        if n_gpu_layers == -1:
            gpu_weight_mb = weight_mb
        else:
            gpu_weight_mb = int(weight_mb * _gpu_layer_fraction(n_gpu_layers, total_layers)) + 256
        kv_mb = protection().llama_kv_cache_reserve_mb(
            path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            total_layers=total_layers,
            weight_mb=weight_mb,
            free_mb=free_mb,
        )
        total_need = gpu_weight_mb + kv_mb
        if weights_resident:
            return max(_MIN_LLAMA_BATCH * 2, free_mb)
        return max(_MIN_LLAMA_BATCH * 2, free_mb - total_need)
    except Exception:
        return free_mb


def _estimate_gguf_params_b(path: Path, weight_mb: int) -> float:
    guessed = guess_params_from_name(path.name) or guess_params_from_name(str(path))
    if guessed:
        return float(guessed)
    # Most local chat GGUFs are Q4/Q5. Inferring params from file size is
    # intentionally conservative because underestimating KV cache causes OOM.
    return max(1.0, float(weight_mb) / 1024.0 / 0.55)


def _gguf_exact_kv_per_token_mb(path: Path) -> float | None:
    """Exact fp16 KV MB/token from GGUF attention metadata, or None."""
    if not path.is_file():
        return None
    try:
        from seiso.inference.backends import gguf_kv_bytes_per_token

        kv_bytes = gguf_kv_bytes_per_token(str(path))
    except Exception:
        return None
    if not kv_bytes:
        return None
    return kv_bytes / (1024**2)


def _llama_effective_kv_ctx(path: Path, n_ctx: int) -> int:
    """Context tokens used for KV sizing (SWA models cap at the sliding window)."""
    ctx = max(_MIN_LLAMA_CTX, min(int(n_ctx), _MAX_LLAMA_CTX))
    if env_bool("SEISO_LLAMA_SWA_FULL", False):
        return ctx
    try:
        from seiso.inference.backends import (
            gguf_sliding_window,
            gguf_swa_layer_fraction,
            gguf_uses_sliding_window_attention,
        )

        if gguf_uses_sliding_window_attention(str(path)):
            sw = gguf_sliding_window(str(path))
            local_ctx = min(ctx, int(sw)) if sw and sw > 0 else min(ctx, 4096)
            swa_frac = gguf_swa_layer_fraction(str(path))
            swa_frac = 0.85 if swa_frac is None else max(0.0, min(float(swa_frac), 1.0))
            global_frac = 1.0 - swa_frac
            return max(
                local_ctx,
                int(swa_frac * local_ctx + global_frac * ctx),
            )
    except Exception:
        pass
    return ctx


def llama_kv_cache_reserve_mb(
    model_path: str | Path,
    *,
    n_ctx: int,
    n_gpu_layers: int,
    total_layers: int | None = None,
    weight_mb: int | None = None,
    free_mb: int = 0,
) -> int:
    """VRAM reserve for llama.cpp KV cache at the requested context.

    Prefers exact GGUF attention geometry (GQA-aware, correct on every NVIDIA
    card); falls back to a conservative parameter-count heuristic when the
    metadata is unavailable.
    """
    if n_gpu_layers == 0:
        return 0
    path = Path(model_path)
    if weight_mb is None:
        weight_mb = int(protection().estimate_path_vram_mb(path))
    if total_layers is None:
        total_layers = gguf_total_layers(path)

    layer_fraction = _gpu_layer_fraction(n_gpu_layers, total_layers)
    ctx = _llama_effective_kv_ctx(path, n_ctx)

    exact_per_token_mb = _gguf_exact_kv_per_token_mb(path)
    if exact_per_token_mb is not None:
        # 10% covers KV padding and per-sequence bookkeeping.
        estimated = int(ctx * exact_per_token_mb * 1.10 * layer_fraction)
        return max(256, estimated)

    params_b = _estimate_gguf_params_b(path, int(weight_mb))
    # Approximate fp16 K+V cache per token. The coefficient tracks observed
    # llama-family/GQA memory by parameter scale while keeping small models fast.
    per_token_mb = max(0.16, min(params_b * 0.045, 3.5))
    estimated = int(ctx * per_token_mb * layer_fraction)
    legacy_floor = max(256, min(int(max(free_mb, 0) * 0.08), 1024))
    return max(legacy_floor, estimated)


def llama_offload_fits_headroom(
    model_path: str | Path,
    *,
    headroom_mb: int,
    n_gpu_layers: int,
    n_ctx: int = 2048,
    weight_mb: int | None = None,
    total_layers: int | None = None,
) -> bool:
    """True when estimated GPU weight + KV for ``n_gpu_layers`` fits within headroom."""
    if n_gpu_layers == 0:
        return True
    if headroom_mb <= 0:
        return False

    path = Path(model_path)
    if weight_mb is None:
        weight_mb = int(protection().estimate_path_vram_mb(path))
    if total_layers is None:
        total_layers = gguf_total_layers(path)

    if n_gpu_layers == -1:
        gpu_weight_mb = weight_mb
    else:
        gpu_weight_mb = int(weight_mb * _gpu_layer_fraction(n_gpu_layers, total_layers)) + 256

    kv_mb = protection().llama_kv_cache_reserve_mb(
        path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        total_layers=total_layers,
        weight_mb=weight_mb,
        free_mb=headroom_mb,
    )
    return gpu_weight_mb + kv_mb <= headroom_mb
