"""Cross-cutting OOM prevention — headroom probes, cache release, and fallbacks."""

from __future__ import annotations

import gc
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any

from seiso.env import env_bool
from seiso.hardware import (
    assess_hardware_fit,
    hardware_profile,
    training_defaults,
    vram_headroom_mb,
)
from seiso.io.files import iter_matching_files
from seiso.memory.estimates import (
    estimate_chat_vram_gb,
    estimate_training_vram_gb,
    guess_params_from_name,
)

logger = logging.getLogger(__name__)

# Reserve a slice of free memory for OS / display / other processes.
_DEFAULT_RESERVE_RATIO = 0.03
# Generation + activations overhead on top of weight estimate.
_INFERENCE_OVERHEAD_MB = 256
_TRAINING_OVERHEAD_RATIO = 2.0
# Absolute ceilings — never exceed even on large machines.
_MAX_INFERENCE_TOKENS = 8192
_MAX_LLAMA_CTX = 131072
_MIN_LLAMA_CTX = 2048
_MAX_LLAMA_BATCH = 2048
_MIN_LLAMA_BATCH = 128
_MAX_LLAMA_CACHE_MB = 1024
_MAX_JSONL_LOAD_MB = 512
_MODEL_WEIGHT_VRAM_SUFFIXES = frozenset({".gguf", ".safetensors", ".bin"})

_VRAM_ESTIMATE_CACHE_MAX = 256
_vram_estimate_cache: dict[tuple, int] = {}
_HEADROOM_CACHE_TTL_S = 2.0
_headroom_cache: tuple[int, float] | None = None


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
    if cache_key is not None and mode == "chat":
        cached = _vram_estimate_cache.get(cache_key)
        if cached is not None:
            return cached

    est = _estimate_path_vram_mb_uncached(p, mode=mode)

    if cache_key is not None and mode == "chat":
        if len(_vram_estimate_cache) >= _VRAM_ESTIMATE_CACHE_MAX:
            _vram_estimate_cache.pop(next(iter(_vram_estimate_cache)))
        _vram_estimate_cache[cache_key] = est
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
        else:
            guessed = (
                guess_params_from_name(name) or guess_params_from_name(path_str) or 7.0
            )
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

    if (
        mode == "train"
        and _hub_model_vram_mb(path_str, mode=mode) is None
        and p.exists()
    ):
        est = int(est * _TRAINING_OVERHEAD_RATIO)
    return max(est, 256)


class MemoryLoadBlockedError(RuntimeError):
    """Raised when a model load would exceed available memory."""


def allow_memory_overcommit() -> bool:
    """When true, log warnings instead of blocking oversized loads."""
    return env_bool("SEISO_ALLOW_MEMORY_OVERCOMMIT", False)


def is_oom_error(exc: BaseException) -> bool:
    """Detect CUDA/MPS/CPU out-of-memory failures across backends."""
    if exc is None:
        return False
    name = type(exc).__name__
    if name in {"OutOfMemoryError", "AcceleratorError"}:
        return True
    msg = str(exc).lower()
    needles = (
        "out of memory",
        "cuda out of memory",
        "mps out of memory",
        "allocat",
        "insufficient memory",
        "failed to allocate",
    )
    return any(n in msg for n in needles)


def release_cached_memory(*, sync: bool = False) -> None:
    """Best-effort GPU/RAM cache release."""
    gc.collect()
    if os.environ.get("SEISO_SKIP_MLX_PROBE", "").strip().lower() not in {
        "1",
        "true",
        "yes",
    }:
        try:
            import mlx.core as mx  # pylint: disable=import-error,no-name-in-module

            if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            if sync:
                torch.cuda.synchronize()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass
    if sync:
        invalidate_headroom_cache()


def llama_batch_headroom_mb(
    free_mb: int,
    *,
    model_path: str | Path | None = None,
    n_gpu_layers: int = -1,
) -> int:
    """VRAM left for llama.cpp batch/KV after estimated weight offload."""
    if not model_path or n_gpu_layers == 0:
        return free_mb
    path = Path(model_path)
    if not path.is_file():
        return free_mb
    try:
        from seiso.inference.backends import gguf_block_count

        weight_mb = int(estimate_path_vram_mb(path))
        total_layers = gguf_block_count(path) or 64
        if n_gpu_layers == -1:
            gpu_weight_mb = weight_mb
        else:
            gpu_fraction = max(0.0, min(float(n_gpu_layers) / float(total_layers), 1.0))
            gpu_weight_mb = int(weight_mb * gpu_fraction) + 256
        if platform.system() == "Linux":
            kv_mb = max(256, min(int(free_mb * 0.08), 1024))
        else:
            try:
                from seiso.inference.model_pool import _gpu_offload_budget_mb

                budget = _gpu_offload_budget_mb()
            except ImportError:
                budget = 0
            kv_pct = 0.05 if budget >= 20 * 1024 else 0.08
            kv_cap = 768 if kv_pct <= 0.05 else 1024
            kv_mb = max(256, min(int(free_mb * kv_pct), kv_cap))
        return max(_MIN_LLAMA_BATCH * 2, free_mb - gpu_weight_mb - kv_mb)
    except Exception:
        return free_mb


def headroom_mb() -> int:
    """Free memory headroom in MB for fit labels and status reporting."""
    global _headroom_cache
    now = time.time()
    if _headroom_cache is not None and now - _headroom_cache[1] < _HEADROOM_CACHE_TTL_S:
        return _headroom_cache[0]
    profile = hardware_profile()
    value: int
    try:
        value = int(vram_headroom_mb(profile))
    except Exception:
        gpus = profile.get("gpus") or []
        if gpus:
            best = 0
            for gpu in gpus:
                total = int(gpu.get("vram_total_mb") or 0)
                used = int(gpu.get("vram_used_mb") or 0)
                if total > 0:
                    best = max(best, max(total - used, 0))
            value = best if best > 0 else _headroom_from_ram(profile)
        else:
            value = _headroom_from_ram(profile)
    _headroom_cache = (value, now)
    return value


def _headroom_from_ram(profile: dict[str, Any]) -> int:
    ram = float(profile.get("ram_gb") or 8)
    try:
        import psutil  # type: ignore

        avail = psutil.virtual_memory().available / (1024**2)
        return int(min(avail * 0.72, ram * 1024 * 0.45))
    except Exception:
        avail = available_ram_mb()
        if avail > 0:
            return int(min(avail * 0.72, ram * 1024 * 0.45))
        return int(ram * 1024 * 0.35)


def invalidate_headroom_cache() -> None:
    """Drop cached headroom (call after unload/load changes VRAM)."""
    global _headroom_cache
    _headroom_cache = None


def available_ram_mb() -> int:
    """Cross-platform available RAM in MB (Linux, macOS, Windows)."""
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available / (1024**2))
    except Exception:
        pass
    if platform.system() == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys / (1024**2))
        except Exception:
            pass
    return int(float(hardware_profile().get("ram_gb") or 8) * 1024 * 0.5)


def build_hf_max_memory(
    *, reserve_ratio: float = _DEFAULT_RESERVE_RATIO
) -> dict[int, str] | None:
    """Build HuggingFace ``max_memory`` unless caps are explicitly disabled."""
    if env_bool("SEISO_DISABLE_MEMORY_CAPS", False):
        return None
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    max_memory: dict[int, str] = {}
    for i in range(torch.cuda.device_count()):
        try:
            free_bytes, _total = torch.cuda.mem_get_info(i)
        except Exception:
            props = torch.cuda.get_device_properties(i)
            free_bytes = int(props.total_memory * (1.0 - reserve_ratio))
        usable = max(int(free_bytes * (1.0 - reserve_ratio)), 256 * 1024**2)
        max_memory[i] = f"{usable // (1024**2)}MiB"
    return max_memory or None


def assess_path_memory_fit(path: str | Path, *, mode: str = "chat") -> dict[str, Any]:
    """Return fit metadata compatible with Forge hardware assessments."""
    est_mb = estimate_path_vram_mb(path, mode=mode)
    est_gb = round(est_mb / 1024, 2)
    profile = hardware_profile()
    try:
        return assess_hardware_fit(est_gb, profile, mode=mode)
    except Exception:
        from seiso.hardware.tiers import fit_headroom_mb, vram_headroom_mb

        capacity = int(fit_headroom_mb(profile))
        free = int(vram_headroom_mb(profile))
        raw_budget = free if free > 0 else capacity
        reserve = max(256, int(raw_budget * 0.02)) if raw_budget > 0 else 0
        budget = max(0, raw_budget - reserve)
        blocked = budget > 0 and est_mb > budget
        budget_gb = round(budget / 1024, 1)
        return {
            "hardware_fit": "unlikely" if blocked else "good",
            "est_vram_mb": est_mb,
            "memory_load_blocked": blocked,
            "memory_load_blocked_reason": (
                f"Needs ~{est_gb:.1f} GB at runtime but only ~{budget_gb} GB is safely available right now."
                if blocked
                else None
            ),
        }


def assess_path_memory_fit_for_load(
    path: str | Path,
    *,
    mode: str = "chat",
    pool: Any | None = None,
    backend: str | None = None,
    unload_if_needed: bool = True,
) -> dict[str, Any]:
    """Assess fit after unloading any active Seiso model that would be replaced."""
    from seiso.inference.model_pool import get_model_pool

    active_pool = pool or get_model_pool()
    if unload_if_needed:
        active_pool.prepare_for_load(str(path), backend)
    fit = assess_path_memory_fit(path, mode=mode)
    if _bypass_apple_llamacpp_preflight_block(fit, backend=backend, mode=mode):
        fit = dict(fit)
        fit["memory_load_blocked"] = False
        fit["memory_load_blocked_reason"] = None
        fit["memory_load_warning"] = (
            "Low free unified memory — trying llama.cpp with mmap plus Mac CPU "
            "offload fallback. Close apps if loading still fails."
        )
    return fit


def _bypass_apple_llamacpp_preflight_block(
    fit: dict[str, Any],
    *,
    backend: str | None,
    mode: str,
) -> bool:
    """Skip only the preflight block; llama.cpp load still handles OOM/fallback."""
    if mode != "chat" or not fit.get("memory_load_blocked"):
        return False
    if str(backend or "").lower() not in {"llamacpp", "llama"}:
        return False
    profile = hardware_profile()
    try:
        from seiso.hardware.tiers import HardwareTier, classify_tier

        return classify_tier(profile) == HardwareTier.APPLE_UNIFIED
    except Exception:
        return False


def ensure_load_fits(
    path: str | Path,
    *,
    mode: str = "chat",
    backend: str | None = None,
) -> dict[str, Any]:
    """Block model loads that exceed measured memory headroom."""
    fit = assess_path_memory_fit_for_load(path, mode=mode, backend=backend)
    if fit.get("memory_load_blocked"):
        reason = (
            fit.get("memory_load_blocked_reason") or "Model exceeds available memory"
        )
        if allow_memory_overcommit():
            logger.warning("Memory overcommit allowed: %s", reason)
        else:
            raise MemoryLoadBlockedError(reason)
    return fit


def _estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    return max(64, int(chars / 3.2))


def sanitize_inference_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Clamp generation limits to available memory without changing intent."""
    if payload.get("_sanitized"):
        return payload
    out = dict(payload)
    messages = out.get("messages") or []
    prompt_tokens = _estimate_prompt_tokens(messages)
    headroom = headroom_mb()
    try:
        from seiso.inference.model_pool import _effective_offload_headroom_mb

        headroom = _effective_offload_headroom_mb(headroom)
    except ImportError:
        pass

    max_tokens = int(out.get("max_tokens") or 2048)
    max_tokens = max(1, min(max_tokens, _MAX_INFERENCE_TOKENS))

    if headroom > _INFERENCE_OVERHEAD_MB:
        kv_budget_tokens = max(
            512,
            int((headroom - _INFERENCE_OVERHEAD_MB) * 128 / 1.15),
        )
        max_tokens = min(max_tokens, max(128, kv_budget_tokens - prompt_tokens - 32))
    out["max_tokens"] = max_tokens

    if out.get("n_ctx") is not None:
        out["n_ctx"] = clamp_llama_n_ctx(
            int(out["n_ctx"]),
            messages=messages,
            max_tokens=max_tokens,
            model_path=out.get("model_path"),
            model_format=out.get("model_format"),
        )
    out["_sanitized"] = True
    return out


def clamp_llama_n_ctx(
    n_ctx: int,
    *,
    messages: list[dict[str, Any]] | None = None,
    max_tokens: int = 512,
    model_path: str | None = None,
    model_format: str | None = None,
    model_name: str | None = None,
) -> int:
    """Bound llama.cpp context to prompt + generation + headroom + model capability."""
    from seiso.inference.context_limits import effective_context_ceiling

    messages = messages or []
    prompt_tokens = _estimate_prompt_tokens(messages)
    needed = prompt_tokens + max_tokens + 128
    step = 512
    sized = ((needed + step - 1) // step) * step
    sized = max(_MIN_LLAMA_CTX, min(_MAX_LLAMA_CTX, sized))

    ctx_cap = effective_context_ceiling(
        model_path,
        model_format=model_format,
        model_name=model_name,
    )

    return min(max(n_ctx, sized), ctx_cap)


def clamp_llama_load_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize llama.cpp load kwargs without RAM/VRAM-based downscaling."""
    out = dict(kwargs)
    out.pop("_model_path", None)
    n_ctx = int(out.get("n_ctx") or _MIN_LLAMA_CTX)

    n_batch = int(out.get("n_batch") or _MAX_LLAMA_BATCH)
    out["n_batch"] = max(_MIN_LLAMA_BATCH, n_batch)
    n_ubatch = int(out.get("n_ubatch") or out["n_batch"])
    out["n_ubatch"] = max(_MIN_LLAMA_BATCH, min(n_ubatch, out["n_batch"]))

    ctx_cap = clamp_llama_n_ctx(n_ctx, max_tokens=512)
    if n_ctx > ctx_cap:
        out["n_ctx"] = ctx_cap
    return out


def clamp_llama_cache_mb(default_mb: int) -> int:
    """Return the configured prompt cache size without memory-based shrinking."""
    return int(default_mb)


def training_pin_memory() -> bool:
    """Pin memory only when CUDA training is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def apply_training_memory_guards(config: Any) -> Any:
    """Keep compatibility fixes while leaving user RAM/VRAM sizing untouched."""
    from seiso.training.config import TrainConfig

    if not isinstance(config, TrainConfig):
        return config

    profile = hardware_profile()
    try:
        defaults = training_defaults(profile)
    except Exception:
        defaults = {
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 2048,
        }

    updates: dict[str, Any] = {}

    # Downgrade quant to the platform-recommended value when the requested mode
    # is unavailable (e.g. QLoRA/4-bit on macOS where bitsandbytes is absent,
    # or 4-bit on a CPU-only box). Without this, torch_loader silently loads a
    # 16-bit model while the trainer still requests paged_adamw_8bit, crashing
    # at optimizer creation with ImportError: bitsandbytes.
    recommended_quant = defaults.get("quant")
    if recommended_quant and str(config.quant) != str(recommended_quant):
        target: Any = None
        try:
            from seiso.training.config import QuantMode

            target = QuantMode(recommended_quant)
        except (ValueError, ImportError):
            target = None
        if target is not None and target != config.quant:
            # Only downgrade — never upgrade beyond what the user asked for.
            rank = {QuantMode.NONE: 0, QuantMode.INT16: 1, QuantMode.INT8: 2, QuantMode.INT4: 3}  # type: ignore[name-defined]
            if rank.get(target, 0) < rank.get(config.quant, 0):
                updates["quant"] = target
                logger.info(
                    "Training memory guards: quant %s -> %s (platform recommendation)",
                    config.quant.value,
                    target.value,
                )

    if not updates:
        return config

    logger.info("Training memory guards applied: %s", updates)
    return config.model_copy(update=updates)


def apply_training_oom_fallback(config: Any) -> Any:
    """Halve batch / seq after an OOM during training."""
    batch = max(1, int(config.batch_size) // 2)
    accum = int(config.gradient_accumulation_steps) * 2
    max_seq = max(128, int(config.max_seq_length) // 2)
    logger.warning(
        "OOM recovery: batch_size=%d accum=%d max_seq_length=%d",
        batch,
        accum,
        max_seq,
    )
    return config.model_copy(
        update={
            "batch_size": batch,
            "gradient_accumulation_steps": accum,
            "max_seq_length": max_seq,
        }
    )


def apply_rl_memory_guards(flat: dict[str, Any]) -> dict[str, Any]:
    """Leave RL quant sizing untouched; OOM fallback handles real failures."""
    out = dict(flat)
    ctx = int(out.get("llama_cpp_context") or 0)
    if ctx > 0:
        out["llama_cpp_context"] = min(ctx, clamp_llama_n_ctx(ctx, max_tokens=512))

    return out


def jsonl_load_safe(path: Path) -> bool:
    """True when JSONL should use datasets loader instead of in-memory list."""
    try:
        return path.stat().st_size > _MAX_JSONL_LOAD_MB * 1024**2
    except OSError:
        return False


def resolve_training_device_map(
    device: str | None = None,
) -> str | dict[str, str] | None:
    """Single-device placement for DDP; auto only for single-process CUDA."""
    import os

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        return {"": f"cuda:{local_rank}"}

    if device == "mps":
        return {"": "mps"}
    try:
        import torch

        if device == "cuda" or (device is None and torch.cuda.is_available()):
            return "auto"
    except ImportError:
        pass
    return None
