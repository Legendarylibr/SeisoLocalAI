"""Isolated GGUF sidecar clients (Ollama-first, llama-swap fallback).

Runtime health/engine/status lives in ``sidecar_runtime``; Ollama model registration
lives in ``ollama_registry``.
"""

from __future__ import annotations

import json
import platform
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from seiso.env import env_bool, env_int, env_str
from seiso.inference.kv_policy import resolve_sidecar_kv_policy
from seiso.inference.sidecar_runtime import (
    DEFAULT_LLAMASWAP_URL,
    DEFAULT_OLLAMA_URL,
    LlamaSwapRuntime,
    SidecarRuntime,
    llamaswap_enabled,
    llamaswap_health_ok,
    llamaswap_setup_hint,
    llamaswap_status,
    llamaswap_url,
    ollama_cli_host,
    ollama_health_ok,
    ollama_url,
    preferred_llamaswap_engine,
    preferred_sidecar_engine,
    sidecar_enabled,
    sidecar_setup_hint,
    sidecar_stack_ready,
    sidecar_status,
)
from seiso.inference.streaming import StreamToken, estimate_chunk_tokens
from seiso.inference.tool_calls import (
    ToolCallDeltaBuffer,
    message_content_with_tool_calls,
)

# Re-export runtime surface (implemented in sidecar_runtime).
__all__ = [
    "DEFAULT_LLAMASWAP_URL",
    "DEFAULT_OLLAMA_URL",
    "IsolatedGgufClient",
    "LlamaSwapClient",
    "LlamaSwapRuntime",
    "OllamaClient",
    "SidecarRuntime",
    "create_isolated_gguf_client",
    "llamaswap_enabled",
    "llamaswap_health_ok",
    "llamaswap_model_name",
    "llamaswap_setup_hint",
    "llamaswap_status",
    "llamaswap_url",
    "ollama_cli_host",
    "ollama_health_ok",
    "ollama_url",
    "plan_sidecar_request",
    "preferred_llamaswap_engine",
    "preferred_sidecar_engine",
    "sidecar_enabled",
    "sidecar_max_tokens",
    "sidecar_num_ctx",
    "sidecar_ollama_keep_alive",
    "sidecar_ollama_num_batch",
    "sidecar_ollama_num_gpu",
    "sidecar_setup_hint",
    "sidecar_stack_ready",
    "sidecar_status",
    "sidecar_vram_context_cap",
]


class IsolatedGgufClient(Protocol):
    engine: str

    def ensure_ready(self) -> None: ...

    def complete(self, payload: dict[str, Any], model_path: str) -> str: ...

    def warm_model(self, payload: dict[str, Any], model_path: str) -> bool: ...

    def stream(
        self,
        payload: dict[str, Any],
        model_path: str,
        *,
        should_stop,
        runtime_stats: dict[str, Any] | None = None,
    ) -> Iterator[StreamToken]: ...

    def release_external_memory(self, model_path: str | None = None) -> tuple[bool, str | None]: ...


def llamaswap_model_name(model_path: str) -> str:
    """Resolve the model identifier expected by the llama-swap OpenAI API."""
    override = env_str("SEISO_LLAMASWAP_MODEL", "").strip()
    if override:
        return override
    return str(Path(model_path).expanduser())


# Headroom kept free between prompt + generation and the context edge.
_SIDECAR_CTX_MARGIN_TOKENS = 256

# Fraction of *free* (not total) VRAM the sidecar may commit to weights + KV.
# Leaves slack for the display/compositor and the engine's compute graph so a
# display-attached consumer GPU cannot be driven into a driver-resetting hang.
_SIDECAR_VRAM_BUDGET_RATIO = 0.55
_SIDECAR_VRAM_BUDGET_RATIO_32GB = 0.65
_SIDECAR_VRAM_BUDGET_RATIO_48GB = 0.70
_SIDECAR_VRAM_BUDGET_RATIO_WORKSTATION = 0.80
_SIDECAR_VRAM_RESERVE_MB = 4096
_SIDECAR_CONSUMER_VRAM_RESERVE_MB = 5120
_SIDECAR_CONSUMER_LARGE_VRAM_RESERVE_MB = 6144
_CONSUMER_NVIDIA_RE = re.compile(r"\brtx\s*(?:20|30|40|50)[5-9]0\b", re.I)
_SIDECAR_CONSUMER_SMALL_FOOTPRINT_RATIO = 0.35
_SIDECAR_CONSUMER_MEDIUM_FOOTPRINT_RATIO = 0.55
_SIDECAR_CONSUMER_LARGE_FOOTPRINT_RATIO = 0.75
_SIDECAR_CONSUMER_SMALL_LAYER_RATIO = 0.50
_SIDECAR_CONSUMER_MEDIUM_LAYER_RATIO = 0.65
_SIDECAR_CONSUMER_LARGE_LAYER_RATIO = 0.80
_SIDECAR_NATIVE_OLLAMA_NUM_BATCH_LOW = 128
_SIDECAR_NATIVE_OLLAMA_NUM_BATCH = 256
_SIDECAR_NATIVE_OLLAMA_NUM_BATCH_ROOMY = 512
_SIDECAR_LOW_HEADROOM_MB = 4096
_SIDECAR_MID_HEADROOM_MB = 8192
_SIDECAR_ROOMY_HEADROOM_MB = 16384
_SIDECAR_NATIVE_OLLAMA_KEEP_ALIVE_LOW = "30s"
_SIDECAR_NATIVE_OLLAMA_KEEP_ALIVE_MID = "1m"
_SIDECAR_NATIVE_OLLAMA_KEEP_ALIVE = "2m"
_SIDECAR_PERF_KEEP_ALIVE = "10m"
_SIDECAR_PERF_VRAM_BUDGET_RATIO = 0.75
_SIDECAR_PERF_VRAM_BUDGET_RATIO_CONSUMER = 0.70


def _sidecar_perf_mode() -> bool:
    """Opt-in higher GPU utilization for sidecar GGUF (default off = safe clamps)."""
    return env_bool("SEISO_SIDECAR_PERF_MODE", False)


def _sidecar_native_linux_nvidia() -> bool:
    try:
        from seiso.inference.backends import _native_linux_requires_isolated_gguf

        return _native_linux_requires_isolated_gguf()
    except Exception:
        # Match native GGUF routing: if Linux detection is broken, do not fail
        # open into an oversized sidecar allocation on a possible NVIDIA host.
        return platform.system() == "Linux"


def _sidecar_vram_clamp_enabled() -> bool:
    return env_bool("SEISO_SIDECAR_VRAM_CLAMP", True)


def _sidecar_consumer_nvidia_gpu() -> bool:
    """True for GeForce/GTX/TITAN and bare RTX xx50-xx90 consumer cards."""
    try:
        from seiso.hardware import hardware_profile

        profile = hardware_profile()
    except Exception:
        return False
    for gpu in profile.get("gpus") or []:
        name = str(gpu.get("name") or "")
        lowered = name.lower()
        if not any(marker in lowered for marker in ("nvidia", "geforce", "rtx", "gtx")):
            continue
        if any(marker in lowered for marker in ("geforce", "gtx", "titan")):
            return True
        if _CONSUMER_NVIDIA_RE.search(name):
            return True
    return False


def _sidecar_vram_budget_ratio() -> float:
    """Adaptive native NVIDIA VRAM budget; env override remains authoritative."""
    raw = env_str("SEISO_SIDECAR_VRAM_BUDGET_RATIO", "").strip()
    if raw:
        try:
            return max(0.50, min(float(raw), 0.95))
        except ValueError:
            pass
    if _sidecar_perf_mode():
        if _sidecar_consumer_nvidia_gpu():
            return _SIDECAR_PERF_VRAM_BUDGET_RATIO_CONSUMER
        return _SIDECAR_PERF_VRAM_BUDGET_RATIO
    if _sidecar_consumer_nvidia_gpu():
        return _SIDECAR_VRAM_BUDGET_RATIO
    try:
        from seiso.memory.protection import discrete_gpu_total_mb

        total_mb = int(discrete_gpu_total_mb())
    except Exception:
        total_mb = 0
    if total_mb <= 0:
        return _SIDECAR_VRAM_BUDGET_RATIO
    if total_mb <= 24 * 1024:
        return _SIDECAR_VRAM_BUDGET_RATIO
    if total_mb <= 32 * 1024:
        return _SIDECAR_VRAM_BUDGET_RATIO_32GB
    if total_mb <= 48 * 1024:
        return _SIDECAR_VRAM_BUDGET_RATIO_48GB
    return _SIDECAR_VRAM_BUDGET_RATIO_WORKSTATION


def _sidecar_vram_reserve_mb(*, total_mb: int = 0, consumer: bool | None = None) -> int:
    """Hard VRAM margin kept outside the sidecar budget."""
    raw = env_str("SEISO_SIDECAR_VRAM_RESERVE_MB", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    if consumer is None:
        consumer = _sidecar_consumer_nvidia_gpu()
    if consumer:
        if total_mb > 24 * 1024:
            return _SIDECAR_CONSUMER_LARGE_VRAM_RESERVE_MB
        return _SIDECAR_CONSUMER_VRAM_RESERVE_MB
    return _SIDECAR_VRAM_RESERVE_MB


def _sidecar_vram_budget_mb(free_mb: int) -> int:
    """Budget for sidecar weights + KV using both ratio and fixed reserve."""
    free = max(0, int(free_mb))
    if free <= 0:
        return 0
    try:
        from seiso.memory.protection import discrete_gpu_total_mb

        total_mb = int(discrete_gpu_total_mb())
    except Exception:
        total_mb = 0
    consumer = _sidecar_consumer_nvidia_gpu()
    ratio_budget = int(free * _sidecar_vram_budget_ratio())
    reserve = _sidecar_vram_reserve_mb(total_mb=total_mb, consumer=consumer)
    reserve_budget = max(0, free - reserve)
    return min(ratio_budget, reserve_budget)


def _sidecar_ollama_manual_layer_ratio() -> float | None:
    """Optional explicit native NVIDIA layer cap for Ollama sidecar acceleration."""
    raw = env_str("SEISO_OLLAMA_GPU_LAYER_RATIO", "").strip()
    if raw:
        try:
            return max(0.0, min(float(raw), 1.0))
        except ValueError:
            pass
    return None


def _sidecar_ollama_manual_layer_cap(total_layers: int) -> int | None:
    total = max(1, int(total_layers))
    ratio = _sidecar_ollama_manual_layer_ratio()
    if ratio is None or ratio >= 1.0:
        return None
    return max(0, min(total, int(total * ratio)))


def _sidecar_ollama_dynamic_layer_cap(
    model_path: str,
    *,
    budget_mb: int,
    weight_mb: int,
    total_layers: int,
    num_ctx: int,
) -> int | None:
    """Consumer NVIDIA offload target based on model footprint and live VRAM budget."""
    if _sidecar_ollama_manual_layer_ratio() is not None:
        return _sidecar_ollama_manual_layer_cap(total_layers)
    manual_cap = _sidecar_ollama_manual_layer_cap(total_layers)
    if not _sidecar_consumer_nvidia_gpu():
        return manual_cap
    budget = max(0, int(budget_mb))
    if budget <= 0:
        return 0
    try:
        from seiso.memory.protection.llama_kv import llama_kv_cache_reserve_mb

        full_kv_mb = int(
            llama_kv_cache_reserve_mb(
                model_path,
                n_ctx=num_ctx,
                n_gpu_layers=-1,
                total_layers=total_layers,
                weight_mb=weight_mb,
                free_mb=budget,
            )
        )
    except Exception:
        # If KV geometry is unavailable, fall back to the fit loop without
        # applying a small-model throttle.
        return manual_cap

    full_need_mb = max(1, int(weight_mb) + full_kv_mb)
    footprint_ratio = full_need_mb / max(1, budget)
    if footprint_ratio <= _SIDECAR_CONSUMER_SMALL_FOOTPRINT_RATIO:
        ratio = _SIDECAR_CONSUMER_SMALL_LAYER_RATIO
    elif footprint_ratio <= _SIDECAR_CONSUMER_MEDIUM_FOOTPRINT_RATIO:
        ratio = _SIDECAR_CONSUMER_MEDIUM_LAYER_RATIO
    elif footprint_ratio <= _SIDECAR_CONSUMER_LARGE_FOOTPRINT_RATIO:
        ratio = _SIDECAR_CONSUMER_LARGE_LAYER_RATIO
    else:
        ratio = 1.0
    dynamic_cap = None if ratio >= 1.0 else max(0, int(max(1, total_layers) * ratio))
    if manual_cap is None:
        return dynamic_cap
    if dynamic_cap is None:
        return manual_cap
    return min(manual_cap, dynamic_cap)


def _sidecar_headroom_mb() -> int:
    try:
        from seiso.memory.protection import headroom_mb

        return max(0, int(headroom_mb()))
    except Exception:
        return 0


def _sidecar_native_max_tokens(max_tokens: int) -> int:
    if not _sidecar_native_linux_nvidia():
        return max_tokens
    if env_bool("SEISO_LLAMA_UNSAFE_LONG_COMPLETIONS", False):
        return max_tokens
    try:
        from seiso.memory.protection import headroom_mb
        from seiso.memory.protection.constants import (
            _NATIVE_LINUX_LOW_HEADROOM_MAX_COMPLETION_TOKENS,
            _NATIVE_LINUX_MAX_COMPLETION_TOKENS,
            _NATIVE_LINUX_PREFILL_CLAMP_MB,
        )

        cap = _NATIVE_LINUX_MAX_COMPLETION_TOKENS
        if headroom_mb() < _NATIVE_LINUX_PREFILL_CLAMP_MB:
            cap = min(cap, _NATIVE_LINUX_LOW_HEADROOM_MAX_COMPLETION_TOKENS)
    except Exception:
        cap = 512
    return max(1, min(int(max_tokens), int(cap)))


def sidecar_max_tokens(max_tokens: int) -> int:
    """Public sidecar-safe completion cap for UI defaults and request planning."""
    return _sidecar_native_max_tokens(max_tokens)


def _sidecar_context_ceiling(payload: dict[str, Any], model_path: str) -> int:
    from seiso.inference.context_limits import effective_context_ceiling

    return effective_context_ceiling(
        model_path,
        model_format=payload.get("model_format"),
        model_name=Path(model_path).name,
    )


def sidecar_vram_context_cap(model_path: str, ceiling: int, *, max_tokens: int = 512) -> int:
    """Bound the sidecar KV context to what free VRAM can hold on native Linux.

    Ollama sizes its KV cache to ``num_ctx`` and places it on the GPU, so an
    oversized context can exhaust VRAM on a display-attached card and hang the
    whole machine (driver reset). This mirrors the in-process llama.cpp guard
    (``native_linux_llama_context_cap``) so the out-of-process path is equally
    safe. Non-native-Linux hosts (macOS, WSL without ack) are returned
    unchanged. Disable via ``SEISO_SIDECAR_VRAM_CLAMP=0``.
    """
    if ceiling <= 0 or not _sidecar_vram_clamp_enabled():
        return ceiling
    if not _sidecar_native_linux_nvidia():
        return ceiling
    try:
        from seiso.memory.protection import headroom_mb
        from seiso.memory.protection.llama_runtime import (
            native_linux_llama_context_cap,
        )

        cap = native_linux_llama_context_cap(
            model_path,
            free_mb=int(headroom_mb()),
            n_gpu_layers=-1,
            ceiling=ceiling,
            max_tokens=max_tokens,
        )
    except Exception:
        if _sidecar_native_linux_nvidia():
            return max(2048, min(int(ceiling), 4096))
        return ceiling
    return max(2048, min(int(ceiling), int(cap)))


def sidecar_ollama_num_gpu(model_path: str, *, num_ctx: int) -> int | None:
    """Layer offload count so Ollama keeps weights + KV within free VRAM.

    Ollama's scheduler estimates offload against *total* VRAM, so on a
    display-attached consumer GPU it can put every layer on the GPU and OOM the
    device (hanging the machine) even though free VRAM was insufficient. Return
    an explicit layer count to force partial CPU offload, or ``None`` to let
    Ollama auto-decide when a full offload comfortably fits. Override with
    ``SEISO_OLLAMA_NUM_GPU`` (``-1`` = all layers); disable the auto guard with
    ``SEISO_SIDECAR_VRAM_CLAMP=0``.
    """
    override = env_str("SEISO_OLLAMA_NUM_GPU", "").strip()
    if override:
        try:
            return int(override)
        except ValueError:
            pass
    if not _sidecar_vram_clamp_enabled() or not _sidecar_native_linux_nvidia():
        return None
    try:
        from seiso.inference.backends import gguf_total_layers
        from seiso.memory.protection import estimate_path_vram_mb, headroom_mb
        from seiso.memory.protection.llama_kv import llama_offload_fits_headroom
    except Exception:
        if _sidecar_native_linux_nvidia():
            return 0
        return None

    free_mb = int(headroom_mb())
    if free_mb <= 0:
        return 0
    budget = _sidecar_vram_budget_mb(free_mb)
    try:
        weight_mb = int(estimate_path_vram_mb(model_path))
        total_layers = max(1, gguf_total_layers(model_path))
        layer_cap = _sidecar_ollama_dynamic_layer_cap(
            model_path,
            budget_mb=budget,
            weight_mb=weight_mb,
            total_layers=total_layers,
            num_ctx=num_ctx,
        )
        if llama_offload_fits_headroom(
            model_path,
            headroom_mb=budget,
            n_gpu_layers=-1,
            n_ctx=num_ctx,
            weight_mb=weight_mb,
            total_layers=total_layers,
        ):
            if layer_cap is not None:
                return layer_cap
            return None
        start_layers = total_layers if layer_cap is None else min(total_layers, layer_cap)
        for layers in range(start_layers, -1, -1):
            if llama_offload_fits_headroom(
                model_path,
                headroom_mb=budget,
                n_gpu_layers=layers,
                n_ctx=num_ctx,
                weight_mb=weight_mb,
                total_layers=total_layers,
            ):
                return layers
    except Exception:
        if _sidecar_native_linux_nvidia():
            return 0
        return None
    return 0


def sidecar_ollama_num_batch() -> int | None:
    """Bound Ollama prompt prefill bursts on native NVIDIA hosts."""
    override = env_str("SEISO_OLLAMA_NUM_BATCH", "").strip()
    if override:
        try:
            value = int(override)
        except ValueError:
            value = 0
        return max(1, value) if value > 0 else None
    if _sidecar_native_linux_nvidia():
        if _sidecar_perf_mode():
            free_mb = _sidecar_headroom_mb()
            if 0 < free_mb < _SIDECAR_LOW_HEADROOM_MB:
                return _SIDECAR_NATIVE_OLLAMA_NUM_BATCH
            return _SIDECAR_NATIVE_OLLAMA_NUM_BATCH_ROOMY
        free_mb = _sidecar_headroom_mb()
        if 0 < free_mb < _SIDECAR_MID_HEADROOM_MB:
            return _SIDECAR_NATIVE_OLLAMA_NUM_BATCH_LOW
        if free_mb >= _SIDECAR_ROOMY_HEADROOM_MB and not _sidecar_consumer_nvidia_gpu():
            return _SIDECAR_NATIVE_OLLAMA_NUM_BATCH_ROOMY
        return _SIDECAR_NATIVE_OLLAMA_NUM_BATCH
    return None


def sidecar_ollama_keep_alive(*, active: bool = False) -> str | None:
    """Keep GPU residency short so idle models release VRAM promptly.

    When ``active`` is True (in-flight chat / preload pin), prefer a longer
    keep_alive so the sidecar does not unload mid-session. Explicit
    ``SEISO_OLLAMA_KEEP_ALIVE`` always wins.
    """
    override = env_str("SEISO_OLLAMA_KEEP_ALIVE", "").strip()
    if override:
        return override
    if active:
        try:
            from seiso.inference.profiles import profile_sidecar_keep_alive_override

            profile_ka = profile_sidecar_keep_alive_override()
            if profile_ka:
                return profile_ka
        except Exception:
            pass
        if _sidecar_perf_mode() or _sidecar_native_linux_nvidia():
            return _SIDECAR_PERF_KEEP_ALIVE
    if _sidecar_native_linux_nvidia():
        if _sidecar_perf_mode():
            free_mb = _sidecar_headroom_mb()
            if 0 < free_mb < _SIDECAR_LOW_HEADROOM_MB:
                return _SIDECAR_NATIVE_OLLAMA_KEEP_ALIVE_MID
            return _SIDECAR_PERF_KEEP_ALIVE
        free_mb = _sidecar_headroom_mb()
        if 0 < free_mb < _SIDECAR_LOW_HEADROOM_MB:
            return _SIDECAR_NATIVE_OLLAMA_KEEP_ALIVE_LOW
        if 0 < free_mb < _SIDECAR_MID_HEADROOM_MB:
            return _SIDECAR_NATIVE_OLLAMA_KEEP_ALIVE_MID
        return _SIDECAR_NATIVE_OLLAMA_KEEP_ALIVE
    return None


def sidecar_num_ctx(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    ceiling: int,
) -> int:
    """Right-size the sidecar context window to prompt + generation.

    Snaps to the same coarse presets as the chat UI so multi-turn history growth
    reuses one loaded KV size instead of forcing a model reload per message. The
    ``ceiling`` passed in is already bounded by both the model's native context
    and (on native Linux NVIDIA) free VRAM via ``sidecar_vram_context_cap`` so
    the KV cache cannot exhaust a display-attached GPU and hang the machine.
    """
    from seiso.inference.context_limits import CONTEXT_WINDOW_PRESETS
    from seiso.memory.protection.chat_guards import _estimate_prompt_tokens

    override = env_int("SEISO_SIDECAR_NUM_CTX", 0)
    if override > 0:
        return max(2048, min(override, ceiling))

    needed = _estimate_prompt_tokens(messages) + max(1, max_tokens) + _SIDECAR_CTX_MARGIN_TOKENS
    for preset in CONTEXT_WINDOW_PRESETS:
        if needed <= preset:
            return min(preset, ceiling)
    return ceiling


def plan_sidecar_request(
    payload: dict[str, Any], model_path: str
) -> tuple[list[dict[str, Any]], int, int]:
    """Return (messages, num_ctx, max_tokens) for a sidecar chat request.

    Long inputs are trimmed (oldest turns first, with an explicit omission
    marker) only when they exceed the model's native context ceiling, so the
    sidecar never silently truncates the prompt from the front and drops the
    system message.

    When ``payload`` carries a pinned ``sidecar_num_ctx`` (from preload), that
    value is reused so multi-turn chat does not resize the sidecar KV window.
    """
    from seiso.memory.protection.chat_guards import trim_llama_messages_to_context

    messages = payload.get("messages") or []
    max_tokens = _sidecar_native_max_tokens(int(payload.get("max_tokens", 512)))
    ceiling = _sidecar_context_ceiling(payload, model_path)
    ceiling = sidecar_vram_context_cap(model_path, ceiling, max_tokens=max_tokens)
    if payload.get("n_ctx"):
        ceiling = min(ceiling, max(2048, int(payload["n_ctx"])))
    pinned = payload.get("sidecar_num_ctx")
    if pinned is not None:
        try:
            num_ctx = max(2048, min(int(pinned), ceiling))
        except (TypeError, ValueError):
            num_ctx = sidecar_num_ctx(messages, max_tokens=max_tokens, ceiling=ceiling)
    else:
        num_ctx = sidecar_num_ctx(messages, max_tokens=max_tokens, ceiling=ceiling)
    messages = trim_llama_messages_to_context(messages, n_ctx=num_ctx, max_tokens=max_tokens)
    return messages, num_ctx, max_tokens


def create_isolated_gguf_client(
    *, url: str | None = None, engine: str | None = None
) -> IsolatedGgufClient:
    """Return OllamaClient when Ollama is the active engine, else LlamaSwapClient."""
    selected = engine or preferred_sidecar_engine()
    if selected == "ollama":
        if not ollama_health_ok():
            raise RuntimeError(
                f"Ollama is not reachable at {ollama_url()}. {sidecar_setup_hint(engine='ollama')}"
            )
        return OllamaClient(url=url)
    return LlamaSwapClient(url=url, engine="llamacpp")


class OllamaClient:
    """Client for Ollama's native /api/chat API.

    Uses the native endpoint (not the OpenAI-compatible one) because only
    /api/chat accepts per-request ``options.num_ctx``. Without it Ollama runs
    at its server default context (typically 4096) and silently truncates
    long prompts from the front.
    """

    def __init__(self, *, url: str | None = None) -> None:
        self.url = (url or ollama_url()).rstrip("/")
        self.engine = "ollama"
        self._pinned_load_plan: dict[str, Any] | None = None

    def ensure_ready(self) -> None:
        if not ollama_health_ok(url=self.url):
            raise RuntimeError(
                f"Ollama is not reachable at {self.url}. {sidecar_setup_hint(engine='ollama')}"
            )

    def complete(self, payload: dict[str, Any], model_path: str) -> str:
        body = self._request_body(payload, model_path, stream=False)
        data = self._post_json("/api/chat", body)
        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, dict):
            return ""
        return message_content_with_tool_calls(message)

    def warm_model(self, payload: dict[str, Any], model_path: str) -> bool:
        """Load weights and pin context without generating user-visible text."""
        _messages, num_ctx, _max_tokens = plan_sidecar_request(payload, model_path)
        options = self._load_options(model_path, num_ctx=num_ctx)
        body: dict[str, Any] = {
            "model": self._resolve_model(model_path, payload),
            "prompt": "",
            "stream": False,
            "options": options,
        }
        keep_alive = sidecar_ollama_keep_alive(active=True)
        if keep_alive:
            body["keep_alive"] = keep_alive
        data = self._post_json("/api/generate", body)
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"Ollama warmup failed: {data['error']}")
        return True

    @property
    def pinned_load_plan(self) -> dict[str, Any]:
        """Immutable load-affecting options for the active Ollama runner."""
        return dict(self._pinned_load_plan or {})

    def _load_options(self, model_path: str, *, num_ctx: int) -> dict[str, Any]:
        pinned = self._pinned_load_plan
        if (
            pinned is not None
            and pinned.get("model_path") == model_path
            and int(pinned.get("num_ctx") or 0) >= num_ctx
        ):
            return {
                key: value
                for key, value in pinned.items()
                if key in {"num_ctx", "num_batch", "num_gpu"}
            }

        options: dict[str, Any] = {"num_ctx": num_ctx}
        num_batch = sidecar_ollama_num_batch()
        if num_batch is not None:
            options["num_batch"] = num_batch
        num_gpu = sidecar_ollama_num_gpu(model_path, num_ctx=num_ctx)
        if num_gpu is not None:
            options["num_gpu"] = num_gpu
        self._pinned_load_plan = {"model_path": model_path, **options}
        return dict(options)

    def stream(
        self,
        payload: dict[str, Any],
        model_path: str,
        *,
        should_stop,
        runtime_stats: dict[str, Any] | None = None,
    ) -> Iterator[StreamToken]:
        body = self._request_body(payload, model_path, stream=True)
        if runtime_stats is not None:
            runtime_stats["sidecar_load_plan"] = self.pinned_load_plan
        req = self._build_request("/api/chat", body)
        tool_buffer = ToolCallDeltaBuffer()
        try:
            with urllib.request.urlopen(req, timeout=None) as response:
                # Native API streams one JSON object per line (not SSE).
                for raw in response:
                    if should_stop():
                        break
                    text = raw.decode("utf-8", errors="replace").strip()
                    if not text:
                        continue
                    try:
                        chunk = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(chunk, dict):
                        continue
                    error = chunk.get("error")
                    if error:
                        raise RuntimeError(f"Ollama error: {error}")
                    message = chunk.get("message") or {}
                    content = message.get("content")
                    if content:
                        text_content = str(content)
                        yield StreamToken(
                            text_content,
                            new_tokens=estimate_chunk_tokens(text_content),
                        )
                    tool_text = tool_buffer.add(message.get("tool_calls"))
                    if tool_text:
                        yield StreamToken(
                            tool_text,
                            new_tokens=estimate_chunk_tokens(tool_text),
                        )
                    if chunk.get("done"):
                        if runtime_stats is not None:
                            for source, target in (
                                ("load_duration", "sidecar_load_ms"),
                                ("prompt_eval_duration", "prefill_ms"),
                                ("eval_duration", "decode_ms"),
                            ):
                                raw_duration = int(chunk.get(source) or 0)
                                if raw_duration > 0:
                                    runtime_stats[target] = round(
                                        raw_duration / 1_000_000.0, 3
                                    )
                            eval_count = int(chunk.get("eval_count") or 0)
                            eval_duration = int(chunk.get("eval_duration") or 0)
                            if eval_count > 0 and eval_duration > 0:
                                runtime_stats["decode_tokens_per_sec"] = round(
                                    eval_count / (eval_duration / 1_000_000_000.0),
                                    3,
                                )
                            runtime_stats["sidecar_resident_confirmed"] = True
                        tool_text = tool_buffer.flush()
                        if tool_text and not should_stop():
                            yield StreamToken(
                                tool_text,
                                new_tokens=estimate_chunk_tokens(tool_text),
                            )
                        break
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama is unavailable at {self.url}. {sidecar_setup_hint(engine='ollama')}"
            ) from exc

    def release_external_memory(self, model_path: str | None = None) -> tuple[bool, str | None]:
        """Best-effort Ollama model unload via keep_alive=0."""
        if not model_path:
            return False, "Ollama unload requires a model path"
        try:
            from seiso.inference.ollama_registry import (
                metadata_for_model_path,
                resolve_ollama_tag,
            )

            meta = metadata_for_model_path(model_path)
            tag = resolve_ollama_tag(model_path, meta)
        except Exception as exc:
            return False, str(exc)
        body = json.dumps({"model": tag, "keep_alive": 0}).encode("utf-8")
        target = urllib.parse.urljoin(f"{self.url}/", "api/generate")
        req = urllib.request.Request(
            target,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30):
                return True, None
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            return False, str(exc)

    def _resolve_model(self, model_path: str, payload: dict[str, Any]) -> str:
        from seiso.inference.ollama_registry import (
            ensure_model_registered,
            metadata_for_model_path,
        )

        meta = metadata_for_model_path(model_path, payload.get("model_metadata"))
        return ensure_model_registered(
            model_path,
            repo_id=meta.get("repo_id") if isinstance(meta.get("repo_id"), str) else None,
            metadata=meta,
            model_format=payload.get("model_format"),
        )

    def _request_body(
        self, payload: dict[str, Any], model_path: str, *, stream: bool
    ) -> dict[str, Any]:
        messages, num_ctx, max_tokens = plan_sidecar_request(payload, model_path)
        options: dict[str, Any] = {
            **self._load_options(model_path, num_ctx=num_ctx),
            "num_predict": max_tokens,
            "temperature": float(payload.get("temperature", 0.0)),
        }
        cache_policy = resolve_sidecar_kv_policy(payload, engine=self.engine)
        if cache_policy.num_keep is not None:
            options["num_keep"] = cache_policy.num_keep
        top_p = payload.get("top_p")
        if top_p is not None:
            options["top_p"] = float(top_p)
        body: dict[str, Any] = {
            "model": self._resolve_model(model_path, payload),
            "messages": messages,
            "stream": stream,
            "options": options,
        }
        # Chat/preload requests pin residency; idle probes use the short default.
        active = bool(payload.get("sidecar_active", True))
        keep_alive = sidecar_ollama_keep_alive(active=active)
        if keep_alive:
            body["keep_alive"] = keep_alive
        tools = payload.get("tools_schemas")
        if tools:
            body["tools"] = tools
        return body

    def _build_request(self, path: str, body: dict[str, Any]) -> urllib.request.Request:
        target = urllib.parse.urljoin(f"{self.url}/", path.lstrip("/"))
        data = json.dumps(body).encode("utf-8")
        return urllib.request.Request(
            target,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = self._build_request(path, body)
        try:
            with urllib.request.urlopen(req, timeout=900) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama is unavailable at {self.url}. {sidecar_setup_hint(engine='ollama')}"
            ) from exc
        return json.loads(raw)


class LlamaSwapClient:
    """Small OpenAI-compatible client for a local llama-swap sidecar."""

    def __init__(self, *, url: str | None = None, engine: str | None = None) -> None:
        self.url = (url or llamaswap_url()).rstrip("/")
        self.engine = engine or "llamacpp"

    def ensure_ready(self) -> None:
        """Verify llama-swap is reachable before preload."""
        if not llamaswap_health_ok(url=self.url):
            raise RuntimeError(
                f"llama-swap is not reachable at {self.url}. "
                f"{sidecar_setup_hint(url=self.url, engine=self.engine)}"
            )

    def complete(self, payload: dict[str, Any], model_path: str) -> str:
        body = self._request_body(payload, model_path, stream=False)
        data = self._post_json("/v1/chat/completions", body)
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return message_content_with_tool_calls(message)

    def warm_model(self, payload: dict[str, Any], model_path: str) -> bool:
        """llama-swap has no portable load endpoint; health is not residency."""
        del payload, model_path
        return False

    def stream(
        self,
        payload: dict[str, Any],
        model_path: str,
        *,
        should_stop,
        runtime_stats: dict[str, Any] | None = None,
    ) -> Iterator[StreamToken]:
        del runtime_stats
        body = self._request_body(payload, model_path, stream=True)
        req = self._build_request("/v1/chat/completions", body)
        tool_buffer = ToolCallDeltaBuffer()
        try:
            with urllib.request.urlopen(req, timeout=None) as response:
                for raw in response:
                    if should_stop():
                        break
                    text = raw.decode("utf-8", errors="replace").strip()
                    if not text or not text.startswith("data:"):
                        continue
                    event = text[5:].strip()
                    if event == "[DONE]":
                        tool_text = tool_buffer.flush()
                        if tool_text and not should_stop():
                            yield StreamToken(
                                tool_text,
                                new_tokens=estimate_chunk_tokens(tool_text),
                            )
                        break
                    try:
                        chunk = json.loads(event)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        text_content = str(content)
                        yield StreamToken(
                            text_content,
                            new_tokens=estimate_chunk_tokens(text_content),
                        )
                    tool_text = tool_buffer.add(delta.get("tool_calls"))
                    if tool_text:
                        yield StreamToken(
                            tool_text,
                            new_tokens=estimate_chunk_tokens(tool_text),
                        )
                    if choices[0].get("finish_reason") == "tool_calls":
                        tool_text = tool_buffer.flush()
                        if tool_text:
                            yield StreamToken(
                                tool_text,
                                new_tokens=estimate_chunk_tokens(tool_text),
                            )
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"llama-swap is unavailable at {self.url}. "
                f"{sidecar_setup_hint(url=self.url, engine=self.engine)}"
            ) from exc

    def _request_body(
        self, payload: dict[str, Any], model_path: str, *, stream: bool
    ) -> dict[str, Any]:
        # llama-server's context size is fixed at launch, so only the trim step
        # of the plan applies here; num_ctx cannot be resized per request.
        # Still attach a non-authoritative hint some llama-swap builds accept.
        messages, num_ctx, max_tokens = plan_sidecar_request(payload, model_path)
        body: dict[str, Any] = {
            "model": llamaswap_model_name(model_path),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": float(payload.get("temperature", 0.0)),
            "stream": stream,
        }
        if env_bool("SEISO_LLAMASWAP_SEND_NUM_CTX", False):
            body["n_ctx"] = num_ctx
        cache_policy = resolve_sidecar_kv_policy(payload, engine=self.engine)
        if cache_policy.cache_prompt is not None:
            body["cache_prompt"] = cache_policy.cache_prompt
        top_p = payload.get("top_p")
        if top_p is not None:
            body["top_p"] = float(top_p)
        tools = payload.get("tools_schemas")
        if tools:
            body["tools"] = tools
        return body

    def _build_request(self, path: str, body: dict[str, Any]) -> urllib.request.Request:
        target = urllib.parse.urljoin(f"{self.url}/", path.lstrip("/"))
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = env_str("SEISO_LLAMASWAP_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return urllib.request.Request(target, data=data, headers=headers, method="POST")

    def release_external_memory(self, model_path: str | None = None) -> tuple[bool, str | None]:
        """Best-effort unload of llama-swap managed processes."""
        scope = env_str("SEISO_LLAMASWAP_UNLOAD_SCOPE", "all").strip().lower()
        if scope in {"0", "false", "none", "disabled"}:
            return False, "llama-swap unload disabled by SEISO_LLAMASWAP_UNLOAD_SCOPE"
        if scope == "model" and model_path:
            model = urllib.parse.quote(llamaswap_model_name(model_path), safe="")
            ok, reason = self._post_management(f"/api/models/unload/{model}")
            if ok:
                return True, None
        return self._post_management("/api/models/unload")

    def _post_management(self, path: str) -> tuple[bool, str | None]:
        target = urllib.parse.urljoin(f"{self.url}/", path.lstrip("/"))
        headers = {"Content-Type": "application/json"}
        api_key = env_str("SEISO_LLAMASWAP_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(
            target,
            data=b"{}",
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30):
                return True, None
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            return False, str(exc)

    def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = self._build_request(path, body)
        try:
            with urllib.request.urlopen(req, timeout=900) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"llama-swap is unavailable at {self.url}. "
                f"{sidecar_setup_hint(url=self.url, engine=self.engine)}"
            ) from exc
        return json.loads(raw)
