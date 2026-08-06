"""llama.cpp context and load-kwarg clamping."""

from __future__ import annotations

import contextlib
import platform
from pathlib import Path
from typing import Any

from seiso import platform as seiso_platform
from seiso.env import env_bool
from seiso.inference.backends import gguf_total_layers
from seiso.inference.family_policy import policy_for_gguf
from seiso.memory.protection._facade import protection
from seiso.memory.protection.chat_guards import _estimate_prompt_tokens, _gguf_has_mmproj_sibling
from seiso.memory.protection.constants import (
    _LLAMA_CTX_BUCKETS,
    _MAX_LLAMA_BATCH,
    _MAX_LLAMA_CTX,
    _MIN_LLAMA_BATCH,
    _MIN_LLAMA_CTX,
    _NATIVE_LINUX_COMPACT_BATCH_FLOOR,
    _NATIVE_LINUX_COMPACT_UBATCH_FLOOR,
    _NATIVE_LINUX_CTX_BUCKETS,
    _NATIVE_LINUX_MMPROJ_RESERVE_MB,
)
from seiso.memory.protection.llama_batch import (
    cap_llama_batch_for_context,
    clamp_llama_batch_pair,
    native_linux_batch_defaults,
)
from seiso.memory.protection.llama_kv import _host_os_reserve_mb
from seiso.memory.protection.llama_runtime import (
    llama_host_batch_headroom_mb,
    native_linux_llama_context_cap,
)


def bucket_llama_n_ctx(needed: int, *, ceiling: int | None = None) -> int:
    """Snap context to coarse buckets so multi-turn chat reuses one loaded KV size."""
    try:
        native_linux = seiso_platform.use_linux_nvidia_inference_guards()
    except Exception:
        native_linux = False
    buckets = _NATIVE_LINUX_CTX_BUCKETS if native_linux else _LLAMA_CTX_BUCKETS
    floor = buckets[0]
    cap = _MAX_LLAMA_CTX if ceiling is None else max(1, int(ceiling))
    need = max(min(floor, cap), int(needed))
    for bucket in buckets:
        if need <= bucket:
            return min(bucket, cap)
    return min(cap, max(min(floor, cap), need))


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

    ctx_cap = effective_context_ceiling(
        model_path,
        model_format=model_format,
        model_name=model_name,
    )
    if model_path:
        ctx_cap = min(
            ctx_cap,
            native_linux_llama_context_cap(
                model_path,
                free_mb=protection().headroom_mb(),
                n_gpu_layers=-1,
                ceiling=ctx_cap,
                max_tokens=max_tokens,
            ),
        )
    sized = bucket_llama_n_ctx(needed, ceiling=ctx_cap)
    try:
        native_linux_nvidia = seiso_platform.use_linux_nvidia_inference_guards()
    except Exception:
        native_linux_nvidia = False
    if native_linux_nvidia and not env_bool("SEISO_LLAMA_UNSAFE_STICKY_CTX", False):
        # On native Linux NVIDIA, a sticky oversized n_ctx keeps extra KV cache
        # resident even after old history has been trimmed. Decay to the bucket
        # the current prompt actually needs unless explicitly overridden.
        return min(max(sized, _MIN_LLAMA_CTX), ctx_cap)
    return min(max(int(n_ctx), sized), ctx_cap)


def clamp_llama_load_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize llama.cpp load kwargs and trim oversized batches near VRAM limits."""
    out = dict(kwargs)
    model_path = out.pop("_model_path", None)
    native_linux_hint = out.pop("_native_linux_nvidia", None)
    max_tokens = int(out.pop("_max_tokens", 512) or 512)
    n_ctx = int(out.get("n_ctx") or _MIN_LLAMA_CTX)
    native_linux_nvidia = False
    if native_linux_hint is not None:
        native_linux_nvidia = bool(native_linux_hint)
    else:
        with contextlib.suppress(Exception):
            native_linux_nvidia = seiso_platform.use_linux_nvidia_inference_guards()
    if native_linux_nvidia:
        default_batch, default_ubatch = native_linux_batch_defaults()
        min_batch = _NATIVE_LINUX_COMPACT_BATCH_FLOOR
        min_ubatch = _NATIVE_LINUX_COMPACT_UBATCH_FLOOR
    else:
        default_batch, default_ubatch = _MAX_LLAMA_BATCH, 1024
        min_batch = _MIN_LLAMA_BATCH
        min_ubatch = _MIN_LLAMA_BATCH
    out["n_batch"] = max(min_batch, int(out.get("n_batch") or default_batch))
    out["n_ubatch"] = max(
        min_ubatch,
        min(int(out.get("n_ubatch") or min(out["n_batch"], default_ubatch)), out["n_batch"]),
    )

    n_gpu_layers = int(out.get("n_gpu_layers") or 0)
    tight = False
    if model_path:
        free_mb = protection().headroom_mb()
        tight = protection().llama_model_is_tight_vram_fit(
            model_path=model_path,
            free_mb=free_mb,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
        )
        host_only = native_linux_nvidia and n_gpu_layers == 0
        if host_only:
            batch_headroom = llama_host_batch_headroom_mb(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                free_vram_mb=free_mb,
            )
            if batch_headroom is None:
                batch_headroom = free_mb
        max_batch, max_ubatch, _tight = protection().resolve_llama_model_batches(
            model_path=model_path,
            free_mb=free_mb,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            load_tier="normal",
            weights_resident=False,
            native_linux_nvidia=native_linux_nvidia,
        )
        if host_only:
            max_batch = min(max_batch, batch_headroom)
            max_ubatch = min(max_ubatch, max_batch)
        else:
            batch_headroom = max_batch
        if native_linux_nvidia and _gguf_has_mmproj_sibling(model_path):
            batch_headroom = max(
                _MIN_LLAMA_BATCH * 2,
                batch_headroom - _NATIVE_LINUX_MMPROJ_RESERVE_MB,
            )
            max_batch = min(max_batch, batch_headroom)
            max_ubatch = min(max_ubatch, max_batch)
        elif not (native_linux_nvidia or tight):
            max_batch, max_ubatch = clamp_llama_batch_pair(_MAX_LLAMA_BATCH, 1024)
        out["n_batch"], out["n_ubatch"] = clamp_llama_batch_pair(
            min(out["n_batch"], max_batch),
            min(out["n_ubatch"], max_ubatch),
            native_linux_nvidia=native_linux_nvidia,
            tight=tight,
            gpu_total_mb=free_mb if native_linux_nvidia else None,
        )
        if (
            native_linux_nvidia
            and out.get("flash_attn")
            and not env_bool("SEISO_LLAMA_UNSAFE_FLASH_ATTN", False)
        ):
            try:
                if model_path and not policy_for_gguf(str(model_path)).allow_flash_attn:
                    out.pop("flash_attn", None)
            except (ImportError, OSError, ValueError):
                pass
        if native_linux_nvidia and tight and n_gpu_layers != 0:
            if not env_bool("SEISO_LLAMA_UNSAFE_FLASH_ATTN", False):
                out.pop("flash_attn", None)
            if not env_bool("SEISO_LLAMA_UNSAFE_OP_OFFLOAD", False):
                out.pop("op_offload", None)
            total_layers = gguf_total_layers(model_path)
            if not env_bool("SEISO_LLAMA_UNSAFE_OP_OFFLOAD", False) and (
                n_gpu_layers == -1 or n_gpu_layers >= total_layers
            ):
                out["offload_kqv"] = False

    elif native_linux_nvidia:
        out["n_batch"], out["n_ubatch"] = clamp_llama_batch_pair(
            out["n_batch"],
            out["n_ubatch"],
            native_linux_nvidia=True,
        )

    ctx_cap = clamp_llama_n_ctx(
        n_ctx,
        max_tokens=max_tokens,
        model_path=str(model_path) if model_path else None,
        model_format="gguf" if model_path else None,
    )
    if n_ctx > ctx_cap:
        out["n_ctx"] = ctx_cap
        n_ctx = ctx_cap
    if native_linux_nvidia:
        out["n_batch"], out["n_ubatch"] = clamp_llama_batch_pair(
            int(out.get("n_batch") or 0),
            int(out.get("n_ubatch") or 0),
            native_linux_nvidia=True,
            tight=tight,
            gpu_total_mb=protection().discrete_gpu_total_mb() or protection().headroom_mb(),
        )
        out["n_batch"], out["n_ubatch"] = cap_llama_batch_for_context(
            int(out.get("n_batch") or 0),
            int(out.get("n_ubatch") or 0),
            n_ctx,
        )
    return out


def clamp_llama_cache_mb(
    default_mb: int,
    *,
    model_path: str | Path | None = None,
) -> int:
    """Cap llama.cpp RAM prompt cache using host memory and model mmap footprint."""
    default_mb = max(0, int(default_mb))
    if default_mb <= 0:
        return 0
    if platform.system() != "Linux":
        return default_mb

    ram_mb = protection().available_ram_mb()
    if ram_mb <= 0:
        return min(default_mb, 512)

    cap = min(default_mb, max(128, ram_mb // 24))
    if model_path and seiso_platform.use_linux_nvidia_inference_guards():
        weight_mb = int(protection().estimate_path_vram_mb(model_path))
        mmap_reserve = max(512, int(weight_mb * 0.12))
        host_budget = max(128, ram_mb - mmap_reserve - _host_os_reserve_mb(ram_mb))
        cap = min(cap, max(0, host_budget // 8))
    return max(0, cap)
