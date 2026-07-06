"""VRAM estimation from model paths."""

from __future__ import annotations

from pathlib import Path

from seiso.io.files import iter_matching_files
from seiso.memory.estimates import (
    estimate_chat_vram_gb,
    estimate_training_vram_gb,
    guess_params_from_name,
)
from seiso.memory.protection.constants import (
    _INFERENCE_OVERHEAD_MB,
    _MODEL_WEIGHT_VRAM_SUFFIXES,
    _TRAINING_OVERHEAD_RATIO,
    _VRAM_ESTIMATE_CACHE_MAX,
    _vram_estimate_cache,
)


def _path_stat_key(p: Path) -> tuple | None:
    try:
        stat = p.stat()
        resolved = str(p.resolve())
        if p.is_file():
            return ("file", resolved, stat.st_mtime, stat.st_size)
        if p.is_dir():
            return ("dir", resolved, stat.st_mtime)
    except OSError:
        return None
    return None


def estimate_path_vram_mb(path: str | Path, *, mode: str = "chat") -> int:
    """Conservative runtime memory estimate from path, size, or name."""
    p = Path(path).expanduser()
    cache_key = _path_stat_key(p)
    if cache_key is not None:
        cached = _vram_estimate_cache.get((cache_key, mode))
        if cached is not None:
            return cached

    est = _estimate_path_vram_mb_uncached(p, mode=mode)

    if cache_key is not None:
        if len(_vram_estimate_cache) >= _VRAM_ESTIMATE_CACHE_MAX:
            _vram_estimate_cache.pop(next(iter(_vram_estimate_cache)))
        _vram_estimate_cache[(cache_key, mode)] = est
    return est


def _hub_model_vram_mb(path_str: str, *, mode: str) -> int | None:
    """VRAM estimate for HuggingFace repo ids (not local paths)."""
    from seiso.models.hub_quant import (
        infer_active_params_b,
        is_hub_model_id,
        is_native_hub_quant_model,
        peek_hub_config,
    )

    if not is_hub_model_id(path_str):
        return None

    config = peek_hub_config(path_str)
    params_b = infer_active_params_b(path_str, config=config)
    label = f"{params_b:g}B"
    native = is_native_hub_quant_model(path_str, config=config, peek=False)
    quant = "mxfp4" if native and mode == "train" else ("Q8_0" if native else "4bit")
    est_gb = (
        estimate_training_vram_gb(label, quant=quant, repo_id=path_str)
        if mode == "train"
        else estimate_chat_vram_gb(label, quant=quant, repo_id=path_str)
    )
    return int(est_gb * 1024)


def _estimate_path_vram_mb_uncached(p: Path, *, mode: str = "chat") -> int:
    name = p.name.lower()
    path_str = str(p)
    from_hub = False

    if not p.exists():
        hub_est = _hub_model_vram_mb(path_str, mode=mode)
        if hub_est is not None:
            return hub_est

    if p.is_file() and p.suffix.lower() in {
        ".gguf",
        ".bin",
        ".safetensors",
        ".pt",
        ".pth",
    }:
        file_mb = max(p.stat().st_size / (1024**2), 1)
        if p.suffix.lower() == ".gguf":
            # GGUF weights map ~1:1 to VRAM when fully offloaded; add KV/activation headroom.
            est = int(file_mb + _INFERENCE_OVERHEAD_MB)
        else:
            est = int(file_mb * 1.15 + _INFERENCE_OVERHEAD_MB)
    elif p.is_dir():
        weight_bytes = 0
        has_gguf = False
        for f in iter_matching_files(p, suffixes=_MODEL_WEIGHT_VRAM_SUFFIXES):
            suffix = f.suffix.lower()
            has_gguf = has_gguf or suffix == ".gguf"
            weight_bytes += f.stat().st_size
        if weight_bytes > 0:
            weight_mb = weight_bytes / (1024**2)
            if has_gguf:
                est = int(weight_mb + _INFERENCE_OVERHEAD_MB)
            else:
                est = int(weight_mb * 1.15 + _INFERENCE_OVERHEAD_MB)
        else:
            guessed = guess_params_from_name(name) or 7.0
            est = int(estimate_chat_vram_gb(f"{guessed}B") * 1024)
    else:
        hub_est = _hub_model_vram_mb(path_str, mode=mode)
        if hub_est is not None:
            est = hub_est
            from_hub = True
        else:
            guessed = guess_params_from_name(name) or guess_params_from_name(path_str) or 7.0
            est = int(estimate_chat_vram_gb(f"{guessed}B", repo_id=path_str) * 1024)
            if mode == "train":
                est = int(
                    estimate_training_vram_gb(
                        f"{guessed}B",
                        quant="4bit",
                        repo_id=path_str,
                    )
                    * 1024
                )

    if mode == "train" and not from_hub and p.exists():
        est = int(est * _TRAINING_OVERHEAD_RATIO)
    return max(est, 256)


