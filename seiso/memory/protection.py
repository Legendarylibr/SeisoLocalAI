"""Cross-cutting OOM prevention — headroom probes, clamps, retries, and load guards."""

from __future__ import annotations

import gc
import logging
import os
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Reserve a slice of free memory for OS / display / other processes.
_DEFAULT_RESERVE_RATIO = 0.10
# Generation + activations overhead on top of weight estimate.
_INFERENCE_OVERHEAD_MB = 768
_TRAINING_OVERHEAD_RATIO = 2.2
# Absolute ceilings — never exceed even on large machines.
_MAX_INFERENCE_TOKENS = 8192
_MAX_LLAMA_CTX = 8192
_MIN_LLAMA_CTX = 2048
_MAX_LLAMA_BATCH = 2048
_MIN_LLAMA_BATCH = 128
_MAX_LLAMA_CACHE_MB = 1024
_MAX_JSONL_LOAD_MB = 512


class MemoryLoadBlockedError(RuntimeError):
    """Raised when a model load would exceed available memory."""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def allow_memory_overcommit() -> bool:
    """When true, log warnings instead of blocking oversized loads."""
    return _env_bool("SEISO_ALLOW_MEMORY_OVERCOMMIT", False)


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
    if os.environ.get("SEISO_SKIP_MLX_PROBE", "").strip().lower() not in {"1", "true", "yes"}:
        try:
            import mlx.core as mx

            if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if sync:
                torch.cuda.synchronize()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def _minimal_hardware_profile() -> dict[str, Any]:
    """Forge-free hardware snapshot for CLI and library-only callers."""
    ram_gb = 8.0
    try:
        import psutil  # type: ignore

        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        pass

    gpus: list[dict[str, Any]] = []
    backend = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            backend = "cuda"
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mb = int(props.total_memory / (1024**2))
                used_mb: int | None = None
                try:
                    free, total = torch.cuda.mem_get_info(i)
                    used_mb = int((total - free) / (1024**2))
                except Exception:
                    pass
                gpus.append(
                    {
                        "index": i,
                        "name": str(props.name),
                        "vram_total_mb": total_mb,
                        "vram_used_mb": used_mb,
                    }
                )
        elif platform.system() == "Darwin":
            try:
                import mlx.core  # noqa: F401

                backend = "mlx"
            except ImportError:
                pass
    except ImportError:
        pass

    return {
        "platform": platform.system().lower(),
        "arch": platform.machine(),
        "backend": backend,
        "ram_gb": ram_gb,
        "gpus": gpus,
        "local_only": True,
    }


def hardware_profile() -> dict[str, Any]:
    """Return cached local hardware profile (Forge when available)."""
    try:
        from forge.services.hardware import hardware_profile as _forge_profile

        return _forge_profile()
    except Exception:
        return _minimal_hardware_profile()


def headroom_mb() -> int:
    """Free memory headroom in MB for fit checks and clamps."""
    profile = hardware_profile()
    try:
        from forge.services.hardware import vram_headroom_mb

        return int(vram_headroom_mb(profile))
    except Exception:
        gpus = profile.get("gpus") or []
        if gpus:
            best = 0
            for gpu in gpus:
                total = int(gpu.get("vram_total_mb") or 0)
                used = int(gpu.get("vram_used_mb") or 0)
                if total > 0:
                    best = max(best, max(total - used, 0))
            if best > 0:
                return best
        ram = float(profile.get("ram_gb") or 8)
        try:
            import psutil  # type: ignore

            avail = psutil.virtual_memory().available / (1024**2)
            return int(min(avail * 0.65, ram * 1024 * 0.4))
        except Exception:
            return int(ram * 1024 * 0.35)


def available_ram_mb() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available / (1024**2))
    except Exception:
        return int(float(hardware_profile().get("ram_gb") or 8) * 1024 * 0.5)


def build_hf_max_memory(*, reserve_ratio: float = _DEFAULT_RESERVE_RATIO) -> dict[int, str] | None:
    """Build HuggingFace ``max_memory`` map from live free VRAM."""
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


def _guess_params_from_name(name: str) -> float | None:
    import re

    m = re.search(r"(\d+(?:\.\d+)?)\s*b", name, re.I)
    return float(m.group(1)) if m else None


def estimate_path_vram_mb(path: str | Path, *, mode: str = "chat") -> int:
    """Conservative runtime memory estimate from path, size, or name."""
    p = Path(path).expanduser()
    name = p.name.lower()

    if p.is_file() and p.suffix.lower() in {".gguf", ".bin", ".safetensors", ".pt", ".pth"}:
        file_mb = max(p.stat().st_size / (1024**2), 1)
        est = int(file_mb * 1.15 + _INFERENCE_OVERHEAD_MB)
    elif p.is_dir():
        weight_bytes = 0
        for pattern in ("*.gguf", "*.safetensors", "*.bin"):
            for f in p.rglob(pattern):
                if f.is_file():
                    weight_bytes += f.stat().st_size
        if weight_bytes > 0:
            est = int(weight_bytes / (1024**2) * 1.15 + _INFERENCE_OVERHEAD_MB)
        else:
            guessed = _guess_params_from_name(name) or 7.0
            try:
                from forge.services.hardware import estimate_chat_vram_gb

                est = int(estimate_chat_vram_gb(f"{guessed}B") * 1024)
            except Exception:
                est = int(guessed * 1024 * 0.7 + _INFERENCE_OVERHEAD_MB)
    else:
        guessed = _guess_params_from_name(name) or _guess_params_from_name(str(path)) or 7.0
        try:
            from forge.services.hardware import estimate_chat_vram_gb

            est = int(estimate_chat_vram_gb(f"{guessed}B") * 1024)
        except Exception:
            est = int(guessed * 1024 * 0.7 + _INFERENCE_OVERHEAD_MB)

    if mode == "train":
        est = int(est * _TRAINING_OVERHEAD_RATIO)
    return max(est, 256)


def assess_path_memory_fit(path: str | Path, *, mode: str = "chat") -> dict[str, Any]:
    """Return fit metadata compatible with Forge hardware assessments."""
    est_mb = estimate_path_vram_mb(path, mode=mode)
    est_gb = round(est_mb / 1024, 2)
    profile = hardware_profile()
    try:
        from forge.services.hardware import assess_hardware_fit

        return assess_hardware_fit(est_gb, profile, mode=mode)
    except Exception:
        free = headroom_mb()
        blocked = free > 0 and est_mb > free
        return {
            "hardware_fit": "unlikely" if blocked else "good",
            "est_vram_mb": est_mb,
            "memory_load_blocked": blocked,
            "memory_load_blocked_reason": (
                f"Needs ~{est_gb:.1f} GB at runtime but only ~{round(free / 1024, 1)} GB is free."
                if blocked
                else None
            ),
        }


def ensure_load_fits(path: str | Path, *, mode: str = "chat") -> dict[str, Any]:
    """Block or warn before loading a model that exceeds headroom."""
    fit = assess_path_memory_fit(path, mode=mode)
    if fit.get("memory_load_blocked"):
        reason = fit.get("memory_load_blocked_reason") or "Model exceeds available memory"
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
    out = dict(payload)
    messages = out.get("messages") or []
    prompt_tokens = _estimate_prompt_tokens(messages)
    headroom = headroom_mb()

    max_tokens = int(out.get("max_tokens") or 512)
    max_tokens = max(1, min(max_tokens, _MAX_INFERENCE_TOKENS))

    # Reserve ~1 MB KV per 128 tokens as a coarse guard.
    kv_budget_tokens = max(256, int((headroom - _INFERENCE_OVERHEAD_MB) * 128 / 4))
    max_tokens = min(max_tokens, max(64, kv_budget_tokens - prompt_tokens - 64))
    out["max_tokens"] = max_tokens

    if out.get("n_ctx") is not None:
        out["n_ctx"] = clamp_llama_n_ctx(int(out["n_ctx"]), messages=messages, max_tokens=max_tokens)
    return out


def clamp_llama_n_ctx(
    n_ctx: int,
    *,
    messages: list[dict[str, Any]] | None = None,
    max_tokens: int = 512,
) -> int:
    """Bound llama.cpp context to prompt + generation + headroom."""
    messages = messages or []
    prompt_tokens = _estimate_prompt_tokens(messages)
    needed = prompt_tokens + max_tokens + 128
    step = 512
    sized = ((needed + step - 1) // step) * step
    sized = max(_MIN_LLAMA_CTX, min(_MAX_LLAMA_CTX, sized))

    headroom = headroom_mb()
    ctx_cap = max(_MIN_LLAMA_CTX, int((headroom - _INFERENCE_OVERHEAD_MB) * 4))
    ctx_cap = min(ctx_cap, _MAX_LLAMA_CTX)
    ctx_cap = (ctx_cap // step) * step or _MIN_LLAMA_CTX

    return min(max(n_ctx, sized), ctx_cap)


def clamp_llama_load_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Scale llama.cpp batch/ubatch down on tight memory."""
    out = dict(kwargs)
    headroom = headroom_mb()
    n_ctx = int(out.get("n_ctx") or _MIN_LLAMA_CTX)

    if headroom < 4096:
        cap_batch = 256
    elif headroom < 8192:
        cap_batch = 512
    elif headroom < 16384:
        cap_batch = 1024
    else:
        cap_batch = _MAX_LLAMA_BATCH

    cap_batch = max(_MIN_LLAMA_BATCH, min(cap_batch, _MAX_LLAMA_BATCH))
    out["n_batch"] = min(int(out.get("n_batch") or cap_batch), cap_batch)
    out["n_ubatch"] = min(int(out.get("n_ubatch") or out["n_batch"]), out["n_batch"])

    # Very large contexts on tight VRAM: prefer CPU layers partial offload safety.
    if headroom < 6144 and int(out.get("n_gpu_layers") or 0) != 0:
        out["n_gpu_layers"] = min(int(out.get("n_gpu_layers") or -1), 24)

    ctx_cap = clamp_llama_n_ctx(n_ctx, max_tokens=512)
    if n_ctx > ctx_cap:
        out["n_ctx"] = ctx_cap
    return out


def clamp_llama_cache_mb(default_mb: int) -> int:
    """Disable or shrink RAM prompt cache when headroom is low."""
    headroom = headroom_mb()
    if headroom < 4096:
        return 0
    if headroom < 8192:
        return min(default_mb, 256)
    if headroom < 12288:
        return min(default_mb, 512)
    return min(default_mb, _MAX_LLAMA_CACHE_MB)


def training_pin_memory() -> bool:
    """Pin memory only when CUDA training is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def apply_training_memory_guards(config: Any) -> Any:
    """Clamp training batch/seq to hardware headroom before the run."""
    from seiso.training.config import TrainConfig

    if not isinstance(config, TrainConfig):
        return config

    profile = hardware_profile()
    try:
        from forge.services.hardware import training_defaults

        defaults = training_defaults(profile)
    except Exception:
        defaults = {
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 2048,
        }

    headroom = headroom_mb()
    updates: dict[str, Any] = {}
    batch = int(config.batch_size)
    accum = int(config.gradient_accumulation_steps)
    max_seq = int(config.max_seq_length)

    if batch > defaults["batch_size"]:
        updates["batch_size"] = defaults["batch_size"]
        batch = defaults["batch_size"]
    if accum < defaults["gradient_accumulation_steps"]:
        updates["gradient_accumulation_steps"] = defaults["gradient_accumulation_steps"]
    if max_seq > defaults["max_seq_length"]:
        updates["max_seq_length"] = defaults["max_seq_length"]
        max_seq = defaults["max_seq_length"]

    est_mb = estimate_path_vram_mb(config.model_id, mode="train")
    if headroom > 0 and est_mb > headroom and batch > 1:
        updates["batch_size"] = 1
        updates["gradient_accumulation_steps"] = max(accum, defaults["gradient_accumulation_steps"]) * 2
    if headroom > 0 and est_mb > int(headroom * 0.85) and max_seq > 1024:
        updates["max_seq_length"] = min(max_seq, 1024)

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
    """Scale RL quant torch settings to free VRAM."""
    out = dict(flat)
    headroom = headroom_mb()
    free_gb = headroom / 1024

    preflight = int(out.get("torch_preflight_batch_size") or 4096)
    if free_gb < 4:
        out["torch_preflight_batch_size"] = min(preflight, 512)
        out["replay_buffer_on_gpu"] = False
    elif free_gb < 8:
        out["torch_preflight_batch_size"] = min(preflight, 2048)
        out["replay_buffer_on_gpu"] = bool(out.get("replay_buffer_on_gpu", True))
    else:
        out["torch_preflight_batch_size"] = preflight

    batch_eps = int(out.get("torch_batch_episodes") or 256)
    if free_gb < 6:
        out["torch_batch_episodes"] = min(batch_eps, 256)
    elif free_gb < 12:
        out["torch_batch_episodes"] = min(batch_eps, 512)

    minibatch = int(out.get("torch_minibatch_size") or 64)
    if free_gb < 6:
        out["torch_minibatch_size"] = min(minibatch, 32)

    replay_cap = int(out.get("replay_buffer_capacity") or 0)
    if replay_cap > 0 and free_gb < 8:
        out["replay_buffer_capacity"] = min(replay_cap, 24_000)

    ctx = int(out.get("llama_cpp_context") or 0)
    if ctx > 0:
        out["llama_cpp_context"] = min(ctx, clamp_llama_n_ctx(ctx, max_tokens=512))

    return out


def estimate_file_ram_mb(path: Path) -> int:
    """Rough RAM needed to load a JSONL file via Python list."""
    size_mb = path.stat().st_size / (1024**2)
    return int(size_mb * 2.5)


def jsonl_load_safe(path: Path) -> bool:
    """True when JSONL should use datasets loader instead of in-memory list."""
    try:
        return path.stat().st_size > _MAX_JSONL_LOAD_MB * 1024**2
    except OSError:
        return False


def run_with_oom_retry(
    fn: Callable[[], T],
    *,
    label: str = "operation",
    on_retry: Callable[[], None] | None = None,
    max_attempts: int = 2,
) -> T:
    """Run ``fn`` once; on OOM release caches, optional ``on_retry``, then retry."""
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if not is_oom_error(exc) or attempt >= max_attempts - 1:
                raise
            logger.warning("%s OOM (attempt %d/%d): %s", label, attempt + 1, max_attempts, exc)
            release_cached_memory(sync=True)
            if on_retry:
                on_retry()
    assert last_exc is not None
    raise last_exc


def resolve_training_device_map(device: str | None = None) -> str | dict[str, str] | None:
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
