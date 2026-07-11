"""Local inference runner — VRAM-managed via ModelPool."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from queue import Empty
from typing import Any, cast

from seiso.env import env_int
from seiso.inference.backends import (
    BACKEND_LLAMASWAP,
    BACKEND_MLX,
    BACKEND_TORCH,
    _native_linux_requires_isolated_gguf,
    is_dflash_draft,
    prepare_model_path,
    resolve_local_backend,
)
from seiso.inference.kv_policy import (
    KVCachePolicy,
    resolve_kv_cache_policy,
    resolve_sidecar_kv_policy,
)
from seiso.inference.model_pool import ModelPool, get_dflash_draft, get_model_pool
from seiso.inference.speculative import (
    DFlashDraftSpeculativeBundle,
    default_num_speculative_tokens,
    iter_speculative_tokens,
    iter_speculative_tokens_dflash,
)
from seiso.inference.stream_bridge import (
    StreamBridgeDone,
    StreamBridgeError,
    ThreadStreamBridge,
)
from seiso.inference.streaming import StreamToken, StreamUpdate
from seiso.inference.tool_calls import ToolCallDeltaBuffer, message_content_with_tool_calls
from seiso.inference.tuning import (
    configure_torch_inference,
    estimate_llama_n_ctx,
    extract_mlx_token_text,
    generate_with_cache_fallback,
    llama_completion_kwargs,
    maybe_compile_torch_decode,
    mlx_stream_kwargs,
    torch_generate_kwargs,
)
from seiso.memory.protection import (
    LlamaLoadTier,
    _estimate_prompt_tokens,
    clamp_llama_n_ctx,
    headroom_mb,
    is_oom_error,
    llama_next_recovery_tier,
    llama_oom_recovery_batch,
    llama_prefill_needs_reload,
    native_linux_batch_defaults,
    release_cached_memory,
    resolve_llama_decode_budget,
    sanitize_inference_payload,
    trim_llama_messages_to_context,
)
from seiso.memory.protection.chat_guards import _trim_message_content_to_token_budget
from seiso.models.chat_format import format_messages_for_prompt

logger = logging.getLogger(__name__)

_MAX_LLAMA_OOM_RECOVERIES = 3
_runner: LocalInferenceRunner | None = None
_runner_lock = threading.Lock()
_inference_executor: Any | None = None
_inference_executor_lock = threading.Lock()


def _get_inference_executor() -> Any:
    """Dedicated single-worker pool for blocking inference (lazy, process-local)."""
    global _inference_executor
    if _inference_executor is not None:
        return _inference_executor
    with _inference_executor_lock:
        if _inference_executor is None:
            from concurrent.futures import ThreadPoolExecutor

            _inference_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="seiso-infer"
            )
        return _inference_executor


def _stream_batch_chars() -> int:
    """Chars to batch after the first token; lower = snappier UI, higher = fewer SSE events."""
    return max(1, env_int("SEISO_STREAM_BATCH_CHARS", 4))


def _torch_generate_with_oom_retry(
    model: Any,
    gen_kwargs: dict[str, Any],
    *,
    retry_on_oom: bool = True,
    can_retry_cache: Callable[[], bool] | None = None,
    prepare_cache_retry: Callable[[], None] | None = None,
) -> Any:
    """Run torch generate once, halving max_new_tokens on OOM."""
    try:
        if can_retry_cache is None and prepare_cache_retry is None:
            return generate_with_cache_fallback(model, gen_kwargs)
        return generate_with_cache_fallback(
            model,
            gen_kwargs,
            can_retry=can_retry_cache,
            prepare_retry=prepare_cache_retry,
        )
    except Exception as exc:
        if not is_oom_error(exc) or not retry_on_oom:
            raise
        release_cached_memory(sync=True)
        reduced = dict(gen_kwargs)
        reduced["max_new_tokens"] = max(
            1,
            int(reduced.get("max_new_tokens", 512)) // 2,
        )
        logger.warning(
            "Torch inference OOM — retrying with max_new_tokens=%s",
            reduced["max_new_tokens"],
        )
        return generate_with_cache_fallback(model, reduced)


def _llama_loaded_batch_fallback() -> tuple[int, int]:
    """Conservative batch pair when a cached llama handle lacks metadata."""
    try:
        from seiso.platform import use_linux_nvidia_inference_guards

        if use_linux_nvidia_inference_guards():
            return native_linux_batch_defaults()
    except ImportError:
        pass
    return 4096, 1024


def _torch_stream_timeout_s() -> int:
    """Poll interval for detecting failed Torch generation threads."""
    return max(1, env_int("SEISO_TORCH_STREAM_TIMEOUT_S", 2))


def _raise_if_dflash_inprocess_blocked(draft_path: str | None) -> None:
    if not draft_path or not is_dflash_draft(draft_path):
        return
    if _native_linux_requires_isolated_gguf():
        raise RuntimeError(
            "dFlash speculative decoding uses an in-process llama.cpp GGUF draft, "
            "which is blocked on native Linux NVIDIA. Disable speculative decoding "
            "or set SEISO_LLAMA_ALLOW_INPROCESS_NATIVE_LINUX=1 to explicitly accept "
            "the in-process llama.cpp risk."
        )


def _llama_n_ctx_for_payload(
    payload: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    model_path: str,
) -> int:
    max_tokens = int(payload.get("max_tokens", 512))
    if payload.get("n_ctx"):
        requested = int(payload["n_ctx"])
        # Treat explicit n_ctx as a user/request cap. Do not grow it to fit old
        # history, because that silently increases KV VRAM; trim messages later.
        sized = clamp_llama_n_ctx(
            requested,
            messages=[],
            max_tokens=max_tokens,
            model_path=model_path,
            model_format=payload.get("model_format"),
        )
        return sized
    return estimate_llama_n_ctx(
        messages,
        max_tokens=max_tokens,
        model_path=model_path,
        model_format=payload.get("model_format"),
    )


def _prepare_llama_messages(
    payload: dict[str, Any],
    model_path: str,
) -> tuple[list[dict[str, Any]], int]:
    """Size context, trim messages, and optionally re-estimate when n_ctx is unset."""
    messages = payload.get("messages", [])
    max_tokens = int(payload.get("max_tokens", 512))
    n_ctx = _llama_n_ctx_for_payload(payload, messages, model_path=model_path)
    messages = trim_llama_messages_to_context(
        messages,
        n_ctx=int(n_ctx),
        max_tokens=max_tokens,
    )
    if not payload.get("n_ctx"):
        n_ctx = estimate_llama_n_ctx(
            messages,
            max_tokens=max_tokens,
            floor=2048,
            model_path=model_path,
            model_format=payload.get("model_format"),
        )
        messages = trim_llama_messages_to_context(
            messages,
            n_ctx=int(n_ctx),
            max_tokens=max_tokens,
        )
    return messages, int(n_ctx)


def _torch_context_limit(model: Any, tokenizer: Any) -> int:
    """Best-effort text-generation context limit before tensors move to CUDA."""
    candidates: list[int] = []
    for raw in (
        getattr(tokenizer, "model_max_length", None),
        getattr(getattr(model, "config", None), "max_position_embeddings", None),
        getattr(getattr(model, "config", None), "max_sequence_length", None),
        getattr(getattr(model, "config", None), "n_positions", None),
    ):
        if raw is None:
            continue
        with contextlib.suppress(TypeError, ValueError):
            value = int(raw)
            if 0 < value < 1_000_000:
                candidates.append(value)
    return min(candidates) if candidates else 4096


def _tokenized_length(tokenizer: Any, prompt: str) -> int:
    encoded = tokenizer(prompt, add_special_tokens=False)
    input_ids = encoded.get("input_ids", []) if isinstance(encoded, dict) else []
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


class _PlainChatFormatter:
    """Minimal formatter sentinel for llama.cpp token counting fallback."""


@dataclass(frozen=True)
class LlamaPromptBudget:
    messages: list[dict[str, Any]]
    max_tokens: int
    input_tokens: int
    context_limit: int


def _llama_prompt_token_length(llm: Any, messages: list[dict[str, Any]]) -> int:
    prompt = format_messages_for_prompt(messages, _PlainChatFormatter())
    tokenize = getattr(llm, "tokenize", None)
    if callable(tokenize):
        for kwargs in ({"add_bos": False, "special": True}, {"add_bos": False}, {}):
            with contextlib.suppress(Exception):
                return len(tokenize(prompt.encode("utf-8"), **kwargs))
    return _estimate_prompt_tokens(messages)


def _fit_llama_messages_to_context(
    llm: Any,
    messages: list[dict[str, Any]],
    *,
    n_ctx: int,
    max_tokens: int,
) -> LlamaPromptBudget:
    """Hard cap llama.cpp prompt by tokenizer count before prefill starts."""
    limit = max(2, int(n_ctx))
    reserve = min(128, max(1, limit // 8))
    clamped_max_tokens = max(1, min(int(max_tokens), limit - reserve - 1))
    prompt_budget = max(1, limit - clamped_max_tokens - reserve)
    trimmed = trim_llama_messages_to_context(
        messages,
        n_ctx=limit,
        max_tokens=clamped_max_tokens,
    )
    current = _llama_prompt_token_length(llm, trimmed)

    while current > prompt_budget and len(trimmed) > 1:
        for idx, message in enumerate(trimmed[:-1]):
            if str(message.get("role", "")).lower() in {"user", "assistant"}:
                trimmed.pop(idx)
                current = _llama_prompt_token_length(llm, trimmed)
                break
        else:
            break

    for _ in range(24):
        if current <= prompt_budget:
            break
        if not trimmed:
            break
        idx = max(
            range(len(trimmed)),
            key=lambda i: _estimate_prompt_tokens([trimmed[i]]),
        )
        content_est = _estimate_prompt_tokens([trimmed[idx]])
        if content_est <= 1:
            break
        ratio = max(0.02, min(0.9, prompt_budget / max(current, 1)))
        target_tokens = max(1, int(content_est * ratio * 0.85))
        updated = dict(trimmed[idx])
        updated["content"] = _trim_message_content_to_token_budget(
            updated.get("content", ""),
            target_tokens,
        )
        if updated.get("content") == trimmed[idx].get("content"):
            break
        trimmed[idx] = updated
        current = _llama_prompt_token_length(llm, trimmed)

    available_tokens = limit - current - reserve
    if available_tokens < 1:
        raise RuntimeError(
            "llama.cpp prompt exceeds context after trimming; reduce prompt size, "
            "knowledge context, or max_tokens"
        )
    clamped_max_tokens = max(1, min(clamped_max_tokens, available_tokens))
    return LlamaPromptBudget(
        messages=trimmed,
        max_tokens=clamped_max_tokens,
        input_tokens=current,
        context_limit=limit,
    )


@dataclass(frozen=True)
class TorchPromptBudget:
    messages: list[dict[str, Any]]
    max_tokens: int
    input_tokens: int
    context_limit: int


def _trim_torch_messages_to_context(
    messages: list[dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    max_tokens: int,
) -> TorchPromptBudget:
    limit = max(2, _torch_context_limit(model, tokenizer))
    reserve = min(8, max(1, limit // 8))
    requested_max_tokens = max(1, int(max_tokens))
    clamped_max_tokens = max(1, min(requested_max_tokens, limit - reserve - 1))
    prompt_budget = max(1, limit - clamped_max_tokens - reserve)
    trimmed = trim_llama_messages_to_context(
        messages,
        n_ctx=limit,
        max_tokens=clamped_max_tokens,
    )
    while len(trimmed) > 1:
        prompt = format_messages_for_prompt(trimmed, tokenizer)
        current = _tokenized_length(tokenizer, prompt)
        if current <= prompt_budget:
            return TorchPromptBudget(
                messages=trimmed,
                max_tokens=clamped_max_tokens,
                input_tokens=current,
                context_limit=limit,
            )
        for idx, message in enumerate(trimmed[:-1]):
            if str(message.get("role", "")).lower() in {"user", "assistant"}:
                trimmed.pop(idx)
                break
        else:
            break

    prompt = format_messages_for_prompt(trimmed, tokenizer)
    current = _tokenized_length(tokenizer, prompt)
    if current <= prompt_budget:
        return TorchPromptBudget(
            messages=trimmed,
            max_tokens=clamped_max_tokens,
            input_tokens=current,
            context_limit=limit,
        )
    trimmed = trim_llama_messages_to_context(
        trimmed,
        n_ctx=prompt_budget + clamped_max_tokens + reserve,
        max_tokens=clamped_max_tokens,
    )
    for _ in range(8):
        prompt = format_messages_for_prompt(trimmed, tokenizer)
        current = _tokenized_length(tokenizer, prompt)
        if current <= prompt_budget:
            break
        if not trimmed:
            break
        last = dict(trimmed[-1])
        content = last.get("content", "")
        if not isinstance(content, str) or len(content) <= 32:
            break
        keep = max(32, int(len(content) * max(0.1, prompt_budget / max(current, 1))))
        last["content"] = content[-keep:]
        trimmed[-1] = last

    prompt = format_messages_for_prompt(trimmed, tokenizer)
    current = _tokenized_length(tokenizer, prompt)
    if current + clamped_max_tokens + reserve > limit:
        clamped_max_tokens = max(1, min(clamped_max_tokens, limit - current - reserve))
        prompt_budget = max(1, limit - clamped_max_tokens - reserve)

    for _ in range(8):
        if current <= prompt_budget:
            break
        if not trimmed:
            break
        last = dict(trimmed[-1])
        content = last.get("content", "")
        if not isinstance(content, str) or len(content) <= 1:
            break
        keep = max(1, int(len(content) * max(0.05, prompt_budget / max(current, 1))))
        last["content"] = content[-keep:]
        trimmed[-1] = last
        prompt = format_messages_for_prompt(trimmed, tokenizer)
        current = _tokenized_length(tokenizer, prompt)

    hard_prompt_budget = max(1, limit - reserve - 1)
    for _ in range(16):
        if current <= hard_prompt_budget:
            break
        if not trimmed:
            break
        last = dict(trimmed[-1])
        content = last.get("content", "")
        if not isinstance(content, str) or len(content) <= 1:
            break
        keep = max(1, int(len(content) * max(0.02, hard_prompt_budget / max(current, 1))))
        if keep >= len(content):
            keep = max(1, len(content) - 1)
        last["content"] = content[-keep:]
        trimmed[-1] = last
        prompt = format_messages_for_prompt(trimmed, tokenizer)
        current = _tokenized_length(tokenizer, prompt)

    available_tokens = limit - current - reserve
    if available_tokens < 1:
        raise RuntimeError(
            "Torch prompt exceeds model context after trimming; reduce prompt size or max_tokens"
        )
    clamped_max_tokens = max(1, min(clamped_max_tokens, available_tokens))
    return TorchPromptBudget(
        messages=trimmed,
        max_tokens=clamped_max_tokens,
        input_tokens=current,
        context_limit=limit,
    )


@dataclass(slots=True)
class _TorchGenerationContext:
    model: Any
    tokenizer: Any
    inputs: dict[str, Any]
    input_len: int
    max_tokens: int
    policy: KVCachePolicy
    payload: dict[str, Any]


class LocalInferenceRunner:
    """Runs chat against local MLX, PyTorch, or llama.cpp with VRAM management."""

    def __init__(self) -> None:
        self._pool = get_model_pool()
        self._last_inference_stats: dict[str, Any] = {}

    @property
    def pool(self) -> ModelPool:
        """Active model pool (public accessor for Forge services)."""
        return self._pool

    @property
    def last_inference_stats(self) -> dict[str, Any]:
        """Additive cache/timing metadata for benchmarks and diagnostics."""
        return dict(self._last_inference_stats)

    def warm_model(self, payload: dict[str, Any]) -> None:
        """Load and warm a model while preventing concurrent unload."""
        with self._pool.inference_lease():
            self._warm_model_impl(payload)

    def _warm_model_impl(self, payload: dict[str, Any]) -> None:
        """Load a model into the pool without generating user-visible text."""
        self._last_inference_stats = {}
        started = time.perf_counter()
        model_path = payload["model_path"]
        route, resolved_path = self._resolve_route(payload, model_path)
        payload = sanitize_inference_payload(payload, isolated=route == "llamaswap")
        if route == "mlx":
            self._pool.get_mlx(resolved_path)
        elif route == "torch":
            import torch

            model, tokenizer = self._pool.get_torch(resolved_path)
            loaded_at = time.perf_counter()
            inputs, _input_len, _max_tokens = self._torch_prepare_inputs(
                model,
                payload.get("messages") or [{"role": "user", "content": "."}],
                tokenizer,
                max_tokens=1,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            warm_started = time.perf_counter()
            warmup_confirmed = True
            warmup_reason: str | None = None
            try:
                with torch.inference_mode():
                    model(**inputs, use_cache=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as exc:
                text = str(exc).lower()
                unsupported = isinstance(exc, TypeError) and any(
                    marker in text
                    for marker in (
                        "unexpected keyword",
                        "use_cache",
                        "forward",
                    )
                )
                if is_oom_error(exc) or not unsupported:
                    raise
                warmup_confirmed = False
                warmup_reason = str(exc)
                logger.warning(
                    "Torch model loaded but eager warmup is unsupported: %s",
                    exc,
                )
            policy = resolve_kv_cache_policy(
                payload,
                model=model,
                input_tokens=int(inputs["input_ids"].shape[-1]),
                max_tokens=1,
                free_mb=headroom_mb(),
            )
            compiled = (
                maybe_compile_torch_decode(model, inputs["input_ids"])
                if policy.compile_decode and warmup_confirmed
                else False
            )
            self._last_inference_stats = {
                "backend": "torch",
                "load_ms": round((loaded_at - started) * 1000.0, 3),
                "warmup_ms": round((time.perf_counter() - warm_started) * 1000.0, 3),
                "load_precision": str(getattr(model, "_seiso_load_precision", "unknown")),
                "attention_implementation": str(
                    getattr(model, "_seiso_attention_implementation", "unknown")
                ),
                "decode_compiled": compiled,
                "resident_confirmed": True,
                "warmup_confirmed": warmup_confirmed,
                "warmup_fallback_reason": warmup_reason,
            }
        elif route == "llamaswap":
            pinned_ctx = payload.get("sidecar_num_ctx") or payload.get("n_ctx")
            try:
                pinned_ctx_i = int(pinned_ctx) if pinned_ctx is not None else None
            except (TypeError, ValueError):
                pinned_ctx_i = None
            if pinned_ctx_i is None:
                try:
                    from seiso.inference.llamaswap import plan_sidecar_request

                    _, planned_ctx, planned_max = plan_sidecar_request(payload, resolved_path)
                    payload = {
                        **payload,
                        "sidecar_num_ctx": planned_ctx,
                        "max_tokens": planned_max,
                        "sidecar_active": True,
                    }
                    pinned_ctx_i = planned_ctx
                except Exception:
                    logger.debug(
                        "Sidecar preload ctx planning failed for %s",
                        resolved_path,
                        exc_info=True,
                    )
            client = self._pool.get_llamaswap(resolved_path, num_ctx=pinned_ctx_i)
            # Eager Ollama registration so first chat is not blocked on `ollama create`.
            if getattr(client, "engine", None) == "ollama":
                try:
                    from seiso.inference.ollama_registry import (
                        ensure_model_registered,
                        metadata_for_model_path,
                    )

                    meta = metadata_for_model_path(resolved_path, payload.get("model_metadata"))
                    ensure_model_registered(
                        resolved_path,
                        repo_id=(
                            meta.get("repo_id") if isinstance(meta.get("repo_id"), str) else None
                        ),
                        metadata=meta,
                        model_format=payload.get("model_format"),
                    )
                except Exception:
                    logger.warning(
                        "Ollama preload registration failed for %s",
                        resolved_path,
                        exc_info=True,
                    )
            warm = getattr(client, "warm_model", None)
            resident_confirmed = bool(warm(payload, resolved_path)) if callable(warm) else False
            self._last_inference_stats = {
                "backend": getattr(client, "engine", "llamaswap"),
                "load_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "resident_confirmed": resident_confirmed,
                "sidecar_num_ctx": pinned_ctx_i,
                "sidecar_load_plan": dict(getattr(client, "pinned_load_plan", {})),
            }
        elif route == "speculative":
            draft_path = payload.get("draft_model_path")
            if not draft_path:
                raise ValueError("draft_model_path required for speculative preload")
            if is_dflash_draft(draft_path):
                self._pool.get_torch(resolved_path, load_in_4bit=True)
                get_dflash_draft(draft_path, n_ctx=self._estimate_dflash_n_ctx(payload, draft_path))
            elif not self._pool.torch_speculative_pair_fits(resolved_path, draft_path):
                logger.warning(
                    "Speculative target+draft pair does not fit current memory; "
                    "preloading target model only"
                )
                self._pool.get_torch(resolved_path, load_in_4bit=True)
            else:
                self._pool.get_torch_speculative(resolved_path, draft_path, load_in_4bit=True)
        else:
            messages, n_ctx = _prepare_llama_messages(payload, resolved_path)
            with self._pool.llama_inference_lease():
                llm = self._pool.get_llama(
                    resolved_path,
                    n_ctx=n_ctx,
                    max_tokens=int(payload.get("max_tokens", 512)),
                )
                if messages:
                    budget = _fit_llama_messages_to_context(
                        llm,
                        messages,
                        n_ctx=n_ctx,
                        max_tokens=int(payload.get("max_tokens", 512)),
                    )
                    messages = budget.messages
                    llm = self._llama_guard_prefill(
                        llm,
                        model_path=resolved_path,
                        messages=messages,
                        n_ctx=n_ctx,
                        prompt_tokens=budget.input_tokens,
                    )

    async def chat(self, payload: dict[str, Any]) -> str:
        loop = asyncio.get_running_loop()
        if payload.get("tools_schemas"):
            model_path = payload.get("model_path") or payload.get("model_id")
            if not model_path:
                raise ValueError("model_path or model_id required")
            route, resolved_path = self._resolve_route(payload, model_path)
            payload = sanitize_inference_payload(payload, isolated=route == "llamaswap")
            if route not in {"llama", "llamaswap"}:
                raise ValueError("Tool calling is only supported with GGUF local backends")
            generation_id = self._pool.bump_generation()
            await self._ensure_model_switch(resolved_path, route=route)
            self._pool.begin_inference()
            try:
                executor = _get_inference_executor()
                if route == "llamaswap":
                    return await loop.run_in_executor(
                        executor,
                        lambda: self._llamaswap_complete(payload, resolved_path, generation_id),
                    )
                return await loop.run_in_executor(
                    executor,
                    lambda: self._llama_complete(payload, resolved_path, generation_id),
                )
            finally:
                self._pool.end_inference()

        model_path = payload.get("model_path") or payload.get("model_id")
        if not model_path:
            raise ValueError("model_path or model_id required")

        route, resolved_path = self._resolve_route(payload, model_path)
        payload = sanitize_inference_payload(payload, isolated=route == "llamaswap")
        generation_id = self._pool.bump_generation()
        await self._ensure_model_switch(
            resolved_path, draft_path=payload.get("draft_model_path"), route=route
        )
        return await loop.run_in_executor(
            _get_inference_executor(),
            lambda: self._complete(payload, resolved_path, route, generation_id),
        )

    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        async for update in self.stream_updates(payload):
            yield update.text

    async def stream_updates(self, payload: dict[str, Any]) -> AsyncIterator[StreamUpdate]:
        model_path = payload.get("model_path") or payload.get("model_id")
        if not model_path:
            raise ValueError("model_path or model_id required")

        route, resolved_path = self._resolve_route(payload, model_path)
        payload = sanitize_inference_payload(payload, isolated=route == "llamaswap")
        draft_path = payload.get("draft_model_path")
        generation_id = self._pool.bump_generation()
        await self._ensure_model_switch(resolved_path, draft_path=draft_path, route=route)

        loop = asyncio.get_running_loop()
        bridge = ThreadStreamBridge(
            loop,
            maxsize=max(1, env_int("SEISO_STREAM_QUEUE_SIZE", 32)),
        )

        def should_stop() -> bool:
            return bridge.cancelled or not self._pool.is_generation_active(generation_id)

        def producer() -> None:
            buffer: list[str] = []
            buffered = 0
            output_tokens = 0
            flushed_once = False
            producer_started = time.perf_counter()
            first_token_at: float | None = None
            batch_chars = _stream_batch_chars()

            def metadata() -> dict[str, Any]:
                now = time.perf_counter()
                if first_token_at is not None:
                    elapsed = max(0.000001, now - first_token_at)
                    self._last_inference_stats["decode_tokens_per_sec"] = round(
                        output_tokens / elapsed, 3
                    )
                return dict(self._last_inference_stats)

            try:
                with self._pool.inference_lease():
                    for part in self._iter_tokens(payload, resolved_path, route, should_stop):
                        if should_stop():
                            break
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                            ttft_ms = (first_token_at - producer_started) * 1000.0
                            self._last_inference_stats["ttft_ms"] = round(ttft_ms, 3)
                            setup_ms = float(
                                self._last_inference_stats.get("load_ms") or 0.0
                            ) + float(self._last_inference_stats.get("tokenize_ms") or 0.0)
                            self._last_inference_stats["ready_to_first_token_ms"] = round(
                                max(0.0, ttft_ms - setup_ms), 3
                            )
                        output_tokens += part.new_tokens
                        buffer.append(part.text)
                        buffered += len(part.text)
                        if not flushed_once:
                            bridge.publish(
                                StreamUpdate(
                                    text="".join(buffer),
                                    output_tokens=output_tokens,
                                    metadata=metadata(),
                                )
                            )
                            buffer.clear()
                            buffered = 0
                            flushed_once = True
                        elif buffered >= batch_chars:
                            bridge.publish(
                                StreamUpdate(
                                    text="".join(buffer),
                                    output_tokens=output_tokens,
                                    metadata=metadata(),
                                )
                            )
                            buffer.clear()
                            buffered = 0
                if buffer and not should_stop():
                    bridge.publish(
                        StreamUpdate(
                            text="".join(buffer),
                            output_tokens=output_tokens,
                            metadata=metadata(),
                        )
                    )
                if (
                    route == "llamaswap"
                    and self._last_inference_stats.get("sidecar_resident_confirmed")
                    and not should_stop()
                ):
                    # Ollama reports load/prefill/decode timings in its final
                    # metadata-only frame, after the last text chunk.
                    bridge.publish(
                        StreamUpdate(
                            text="",
                            output_tokens=output_tokens,
                            metadata=metadata(),
                        )
                    )
            except Exception as exc:
                if buffer and not should_stop():
                    bridge.publish(
                        StreamUpdate(
                            text="".join(buffer),
                            output_tokens=output_tokens,
                            metadata=metadata(),
                        )
                    )
                if not should_stop():
                    bridge.publish(StreamBridgeError(exc))
            finally:
                bridge.producer_finished()

        threading.Thread(target=producer, daemon=True).start()

        try:
            while True:
                if should_stop():
                    break
                item = await bridge.next()
                if isinstance(item, StreamBridgeDone):
                    break
                if isinstance(item, StreamBridgeError):
                    raise item.exc
                yield item
        finally:
            bridge.cancel()
            await bridge.wait_for_producer(
                max(0.0, env_int("SEISO_STREAM_STOP_WAIT_MS", 1000) / 1000.0)
            )

    async def cancel_and_unload(self) -> dict:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._pool.cancel_and_unload)
        return self._pool.status()

    async def cancel_generation(self) -> dict:
        """Stop active streams without unloading the warmed model."""
        from seiso.inference.torch_stream import clear_torch_prefix_cache

        self._pool.bump_generation()
        clear_torch_prefix_cache()
        return self._pool.status()

    async def _ensure_model_switch(
        self, model_path: str, *, draft_path: str | None = None, route: str = "llama"
    ) -> None:
        status = self._pool.status()
        active_path = status.get("path")
        active_draft = status.get("draft_path")
        loop = asyncio.get_running_loop()
        if draft_path:
            if (
                active_path
                and active_draft
                and self._pool.normalize_path(active_path) == self._pool.normalize_path(model_path)
                and self._pool.normalize_path(active_draft) == self._pool.normalize_path(draft_path)
            ):
                return
            if not is_dflash_draft(draft_path):
                # Torch+Torch speculative bundles are distinct pool handles; unload
                # a warmed single target before loading target+draft together.
                await loop.run_in_executor(None, lambda: self._pool.prepare_for_load())
                return
            # dFlash reuses the torch target handle; drop torch+torch bundles first.
            active_key = status.get("active_model") or ""
            if active_key.startswith("spec:"):
                await loop.run_in_executor(None, lambda: self._pool.prepare_for_load())
            else:
                await loop.run_in_executor(
                    None,
                    lambda: self._pool.prepare_for_load(model_path, BACKEND_TORCH),
                )
            return

        if active_draft:
            # Turning off speculative decoding must drop the target+draft bundle.
            await loop.run_in_executor(None, lambda: self._pool.prepare_for_load())
            return

        backend = BACKEND_LLAMASWAP if route == "llamaswap" else None
        if self._pool.would_switch_model(model_path, backend):
            await loop.run_in_executor(
                None, lambda: self._pool.prepare_for_load(model_path, backend)
            )

    def _resolve_route(self, payload: dict[str, Any], model_path: str) -> tuple[str, str]:
        if payload.get("draft_model_path"):
            _raise_if_dflash_inprocess_blocked(payload.get("draft_model_path"))
            # dflash drafts are fast GGUF; we still run verification on torch target path for now
            resolved = prepare_model_path(model_path, BACKEND_TORCH)
            return "speculative", resolved

        backend = resolve_local_backend(
            model_path=model_path,
            model_format=payload.get("model_format"),
            requested=payload.get("inference_backend"),
        )
        resolved = prepare_model_path(model_path, backend)
        if backend == BACKEND_MLX:
            return "mlx", resolved
        if backend == BACKEND_TORCH:
            return "torch", resolved
        if backend == BACKEND_LLAMASWAP:
            return "llamaswap", resolved
        return "llama", resolved

    @staticmethod
    def _estimate_dflash_n_ctx(payload: dict[str, Any], draft_path: str) -> int:
        return _llama_n_ctx_for_payload(
            {**payload, "model_format": "gguf"},
            payload.get("messages") or [],
            model_path=draft_path,
        )

    def _iter_tokens(
        self,
        payload: dict[str, Any],
        model_path: str,
        route: str,
        should_stop: Callable[[], bool],
    ) -> Iterator[StreamToken]:
        if route == "speculative":
            yield from self._torch_speculative_stream(payload, model_path, should_stop)
        elif route == "mlx":
            yield from self._mlx_stream(payload, model_path, should_stop)
        elif route == "torch":
            yield from self._torch_stream(payload, model_path, should_stop)
        elif route == "llamaswap":
            yield from self._llamaswap_stream(payload, model_path, should_stop)
        else:
            yield from self._llama_stream(payload, model_path, should_stop)

    def _complete(
        self,
        payload: dict[str, Any],
        model_path: str,
        route: str,
        generation_id: int,
    ) -> str:
        with self._pool.inference_lease():
            if route == "speculative":
                chunks: list[str] = []

                def should_stop() -> bool:
                    return not self._pool.is_generation_active(generation_id)

                for token in self._torch_speculative_stream(payload, model_path, should_stop):
                    if should_stop():
                        break
                    chunks.append(token.text)
                return "".join(chunks)
            if route == "mlx":
                return self._mlx_complete(payload, model_path, generation_id)
            if route == "torch":
                return self._torch_complete(payload, model_path, generation_id)
            if route == "llamaswap":
                return self._llamaswap_complete(payload, model_path, generation_id)
            return self._llama_complete(payload, model_path, generation_id)

    def _torch_speculative_stream(
        self,
        payload: dict[str, Any],
        model_path: str,
        should_stop: Callable[[], bool],
    ) -> Iterator[StreamToken]:
        draft_path = payload.get("draft_model_path")
        if not draft_path:
            raise ValueError("draft_model_path required for speculative decoding")
        _raise_if_dflash_inprocess_blocked(draft_path)

        temperature = float(payload.get("temperature", 0.0))
        if temperature > 0:
            # The custom verifier implements greedy acceptance only. Falling
            # back preserves the requested sampling distribution instead of
            # silently turning temperature sampling into deterministic argmax.
            logger.info(
                "Speculative decoding supports greedy generation only; "
                "using target-only Torch sampling"
            )
            yield from self._torch_stream(payload, model_path, should_stop)
            return
        if not is_dflash_draft(draft_path) and not self._pool.torch_speculative_pair_fits(
            model_path, draft_path
        ):
            logger.warning(
                "Speculative target+draft pair does not fit current memory; "
                "using target-only Torch generation"
            )
            yield from self._torch_stream(payload, model_path, should_stop)
            return

        configure_torch_inference()

        if is_dflash_draft(draft_path):
            # dFlash speculative: fast GGUF dflash draft (llama.cpp) + target verifier (torch)
            target_model, target_tok = self._pool.get_torch(model_path, load_in_4bit=True)
            draft_llm = get_dflash_draft(
                draft_path, n_ctx=self._estimate_dflash_n_ctx(payload, draft_path)
            )

            dflash_bundle = DFlashDraftSpeculativeBundle(
                target_model=target_model,
                target_tokenizer=target_tok,
                draft_llm=draft_llm,
                draft_tokenizer=target_tok,  # vocab alignment expected for dflash
            )

            budget = _trim_torch_messages_to_context(
                payload.get("messages", []),
                model=target_model,
                tokenizer=target_tok,
                max_tokens=int(payload.get("max_tokens", 512)),
            )
            messages = budget.messages
            prompt = format_messages_for_prompt(messages, target_tok)
            max_new_tokens = budget.max_tokens
            num_speculative_tokens = default_num_speculative_tokens(payload)

            yield from iter_speculative_tokens_dflash(
                bundle=dflash_bundle,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                num_speculative_tokens=num_speculative_tokens,
                temperature=temperature,
                should_stop=should_stop,
            )
            return

        # Original torch + torch speculative
        torch_bundle = self._pool.get_torch_speculative(model_path, draft_path, load_in_4bit=True)
        budget = _trim_torch_messages_to_context(
            payload.get("messages", []),
            model=torch_bundle.target_model,
            tokenizer=torch_bundle.target_tokenizer,
            max_tokens=int(payload.get("max_tokens", 512)),
        )
        messages = budget.messages
        prompt = format_messages_for_prompt(messages, torch_bundle.target_tokenizer)
        max_new_tokens = budget.max_tokens
        num_speculative_tokens = default_num_speculative_tokens(payload)

        emitted = False
        try:
            for token in iter_speculative_tokens(
                bundle=torch_bundle,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                num_speculative_tokens=num_speculative_tokens,
                temperature=temperature,
                should_stop=should_stop,
            ):
                emitted = True
                yield token
        except Exception as exc:
            if not is_oom_error(exc):
                raise
            if emitted:
                raise RuntimeError(
                    "Speculative inference OOM after streaming began — "
                    "aborting instead of replaying partial output"
                ) from exc
            release_cached_memory(sync=True)
            reduced = max(1, max_new_tokens // 2)
            logger.warning("Speculative inference OOM — retrying with max_new_tokens=%s", reduced)
            yield from iter_speculative_tokens(
                bundle=torch_bundle,
                prompt=prompt,
                max_new_tokens=reduced,
                num_speculative_tokens=max(1, num_speculative_tokens // 2),
                temperature=temperature,
                should_stop=should_stop,
            )

    def _mlx_stream(
        self,
        payload: dict[str, Any],
        model_path: str,
        should_stop: Callable[[], bool],
    ) -> Iterator[StreamToken]:
        try:
            from mlx_lm import generate
        except ImportError as exc:
            raise RuntimeError("MLX not available — install mlx-lm on macOS") from exc

        model, tokenizer = self._pool.get_mlx(model_path)
        prompt = format_messages_for_prompt(payload.get("messages", []), tokenizer)
        gen_kwargs = {"prompt": prompt, **mlx_stream_kwargs(payload)}

        try:
            from mlx_lm import stream_generate

            for token in stream_generate(model, tokenizer, **gen_kwargs):
                if should_stop():
                    break
                text = extract_mlx_token_text(token)
                if text:
                    yield StreamToken(text)
            return
        except (ImportError, TypeError):
            pass

        if not should_stop():
            yield StreamToken(generate(model, tokenizer, **gen_kwargs))

    def _mlx_complete(
        self,
        payload: dict[str, Any],
        model_path: str,
        generation_id: int,
    ) -> str:
        try:
            from mlx_lm import generate
        except ImportError as exc:
            raise RuntimeError("MLX not available — install mlx-lm on macOS") from exc

        model, tokenizer = self._pool.get_mlx(model_path)
        prompt = format_messages_for_prompt(payload.get("messages", []), tokenizer)
        text = generate(model, tokenizer, prompt=prompt, **mlx_stream_kwargs(payload))
        if not self._pool.is_generation_active(generation_id):
            return ""
        return str(text)

    @staticmethod
    def _torch_prepare_inputs(
        model: Any,
        messages: list[dict[str, Any]],
        tokenizer: Any,
        *,
        max_tokens: int,
    ) -> tuple[dict[str, Any], int, int]:
        budget = _trim_torch_messages_to_context(
            messages,
            model=model,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
        )
        messages = budget.messages
        prompt = format_messages_for_prompt(messages, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt")
        device = LocalInferenceRunner._torch_input_device(model)
        moved = {
            k: v.to(device, non_blocking=getattr(device, "type", "") == "cuda")
            for k, v in inputs.items()
        }
        return moved, int(moved["input_ids"].shape[-1]), budget.max_tokens

    @staticmethod
    def _torch_input_device(model: Any) -> Any:
        device_map = getattr(model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for raw_device in device_map.values():
                device = LocalInferenceRunner._normalize_torch_device(raw_device)
                if device is not None and device.type not in {"cpu", "meta"}:
                    return device
        device = getattr(model, "device", None)
        if device is not None:
            return device
        return next(model.parameters()).device

    @staticmethod
    def _normalize_torch_device(raw_device: Any) -> Any | None:
        import torch

        if raw_device is None:
            return None
        if isinstance(raw_device, torch.device):
            return raw_device
        if isinstance(raw_device, int):
            return torch.device(f"cuda:{raw_device}")
        text = str(raw_device).strip().lower()
        if not text or text in {"disk", "offload"}:
            return None
        if text.isdigit():
            return torch.device(f"cuda:{text}")
        try:
            return torch.device(text)
        except (TypeError, RuntimeError):
            logger.debug("Ignoring unrecognized torch device map entry: %r", raw_device)
            return None

    def _prepare_torch_generation(
        self,
        payload: dict[str, Any],
        model_path: str,
    ) -> _TorchGenerationContext:
        configure_torch_inference()
        status_before = self._pool.status()
        resident_before = (
            status_before.get("path") is not None
            and self._pool.normalize_path(str(status_before["path"]))
            == self._pool.normalize_path(model_path)
            and str(status_before.get("backend") or "").lower() == "torch"
        )
        load_started = time.perf_counter()
        model, tokenizer = self._pool.get_torch(model_path)
        loaded_at = time.perf_counter()
        inputs, input_len, max_tokens = self._torch_prepare_inputs(
            model,
            payload.get("messages", []),
            tokenizer,
            max_tokens=int(payload.get("max_tokens", 512)),
        )
        policy = resolve_kv_cache_policy(
            payload,
            model=model,
            input_tokens=int(inputs["input_ids"].shape[-1]),
            max_tokens=max_tokens,
            free_mb=headroom_mb(),
        )
        self._last_inference_stats = {
            "backend": "torch",
            "cache_mode": policy.cache_implementation,
            "cache_fallback": policy.fallback_reason,
            "estimated_cache_mb": policy.estimated_cache_mb,
            "peak_headroom_mb": policy.headroom_mb,
            "resident_before": resident_before,
            "load_ms": round((loaded_at - load_started) * 1000.0, 3),
            "tokenize_ms": round((time.perf_counter() - loaded_at) * 1000.0, 3),
            "load_precision": str(getattr(model, "_seiso_load_precision", "unknown")),
            "attention_implementation": str(
                getattr(model, "_seiso_attention_implementation", "unknown")
            ),
        }
        self._last_inference_stats["decode_compiled"] = (
            maybe_compile_torch_decode(model, inputs["input_ids"])
            if policy.compile_decode
            else False
        )
        resolved_payload = {
            **payload,
            "max_tokens": max_tokens,
            "cache_implementation": policy.cache_implementation,
            "kv_policy": policy.as_dict(),
        }
        return _TorchGenerationContext(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
            input_len=input_len,
            max_tokens=max_tokens,
            policy=policy,
            payload=resolved_payload,
        )

    def _torch_complete(
        self,
        payload: dict[str, Any],
        model_path: str,
        generation_id: int,
    ) -> str:
        import torch

        context = self._prepare_torch_generation(payload, model_path)
        gen_kwargs = torch_generate_kwargs(
            context.payload,
            context.inputs,
            streamer=None,
            pad_token_id=context.tokenizer.pad_token_id,
        )
        gen_kwargs.pop("streamer", None)

        with torch.inference_mode():
            generated = _torch_generate_with_oom_retry(context.model, gen_kwargs)

        if not self._pool.is_generation_active(generation_id):
            return ""
        output_ids = generated[0][context.input_len :]
        return str(context.tokenizer.decode(output_ids, skip_special_tokens=True))

    def _torch_stream(
        self,
        payload: dict[str, Any],
        model_path: str,
        should_stop: Callable[[], bool],
    ) -> Iterator[StreamToken]:
        from seiso.inference.torch_stream import (
            iter_torch_kv_tokens,
            use_manual_torch_kv_stream,
        )

        context = self._prepare_torch_generation(payload, model_path)
        model = context.model
        tokenizer = context.tokenizer
        inputs = context.inputs
        payload = context.payload
        policy = context.policy

        if use_manual_torch_kv_stream(payload) and policy.manual_stream_compatible:
            emitted = False
            try:
                for token in iter_torch_kv_tokens(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=inputs["input_ids"],
                    max_new_tokens=int(payload.get("max_tokens", 512)),
                    temperature=float(payload.get("temperature", 0.0)),
                    top_p=(float(payload["top_p"]) if payload.get("top_p") is not None else None),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=getattr(tokenizer, "eos_token_id", None),
                    should_stop=should_stop,
                    prefill_chunk_size=policy.prefill_chunk_size,
                    cache_key=f"{self._pool.normalize_path(model_path)}:{id(model)}",
                    prefix_cache=policy.prefix_cache,
                    stats=self._last_inference_stats,
                ):
                    emitted = True
                    yield token
                return
            except Exception as exc:
                if is_oom_error(exc):
                    raise
                if emitted:
                    raise RuntimeError(
                        "Manual Torch KV stream failed after streaming began — "
                        "aborting instead of replaying partial output"
                    ) from exc
                logger.debug(
                    "Manual torch KV stream unavailable — falling back to generate: %s",
                    exc,
                )

        yield from self._torch_stream_generate(model, tokenizer, inputs, payload, should_stop)

    def _torch_stream_generate(
        self,
        model: Any,
        tokenizer: Any,
        inputs: dict[str, Any],
        payload: dict[str, Any],
        should_stop: Callable[[], bool],
    ) -> Iterator[StreamToken]:
        """HF TextIteratorStreamer path (non-cooperative cancel)."""
        from transformers import TextIteratorStreamer

        class _ReplaySafeStreamer(TextIteratorStreamer):
            generated_puts = 0

            def put(self, value: Any) -> None:
                if not self.next_tokens_are_prompt:
                    self.generated_puts += 1
                super().put(value)

            def reset_before_retry(self) -> None:
                self.next_tokens_are_prompt = True
                self.token_cache.clear()
                self.print_len = 0

        streamer = _ReplaySafeStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=_torch_stream_timeout_s(),
        )
        gen_kwargs = torch_generate_kwargs(
            payload,
            inputs,
            streamer,
            pad_token_id=tokenizer.pad_token_id,
        )
        generation_errors: list[BaseException] = []

        def _generate() -> None:
            import torch

            with torch.inference_mode():
                try:
                    # Retrying with the same streamer can duplicate text if an
                    # OOM occurs after partial output has been delivered.
                    _torch_generate_with_oom_retry(
                        model,
                        gen_kwargs,
                        retry_on_oom=False,
                        can_retry_cache=lambda: streamer.generated_puts == 0,
                        prepare_cache_retry=streamer.reset_before_retry,
                    )
                except Exception as exc:
                    generation_errors.append(exc)
                    with contextlib.suppress(Exception):
                        streamer.on_finalized_text("", stream_end=True)

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()
        try:
            while True:
                if should_stop():
                    break
                if generation_errors:
                    raise generation_errors[0]
                try:
                    text = next(streamer)
                except StopIteration:
                    break
                except Empty:
                    if not thread.is_alive():
                        if generation_errors:
                            raise generation_errors[0] from None
                        break
                    continue
                if text:
                    yield StreamToken(text)
        finally:
            # HF generate is not cooperatively cancellable. Wait for the
            # worker to finish so pool unload cannot free the model under it.
            deadline = time.time() + max(_torch_stream_timeout_s(), 600.0)
            while thread.is_alive() and time.time() < deadline:
                thread.join(timeout=0.5)
            if thread.is_alive():
                logger.warning(
                    "Torch generate thread still running after cancel wait — "
                    "deferring pool release until it exits"
                )
                thread.join()
        if generation_errors and not should_stop():
            raise generation_errors[0]

    def _llama_handle_tier(self, llm: Any) -> LlamaLoadTier:
        return getattr(llm, "_seiso_load_tier", "normal") or "normal"

    def _llama_recover_from_oom(
        self,
        llm: Any,
        *,
        model_path: str,
        n_ctx: int,
        max_tokens: int,
    ) -> Any:
        current = self._llama_handle_tier(llm)
        next_tier = llama_next_recovery_tier(current)
        if next_tier is None:
            raise RuntimeError("llama.cpp inference OOM — recovery tiers exhausted")
        batch_override = llama_oom_recovery_batch(
            safe_batch=int(getattr(llm, "_seiso_last_safe_batch", 0) or 0),
            safe_ubatch=int(getattr(llm, "_seiso_last_safe_ubatch", 0) or 0),
            loaded_batch=int(getattr(llm, "_seiso_n_batch", 0) or 0),
            loaded_ubatch=int(getattr(llm, "_seiso_n_ubatch", 0) or 0),
            next_tier=next_tier,
        )
        logger.warning(
            "llama.cpp inference OOM at tier=%s — reloading at tier=%s",
            current,
            next_tier,
        )
        recovery_ctx = min(int(n_ctx), 4096 if next_tier == "compact" else 2048)
        release_cached_memory(sync=True)
        return self._pool.reload_llama(
            model_path,
            recovery_ctx,
            tier=next_tier,
            batch_override=batch_override,
            max_tokens=max_tokens,
        )

    def _llama_guard_prefill(
        self,
        llm: Any,
        *,
        model_path: str,
        messages: list[dict[str, Any]],
        n_ctx: int,
        prompt_tokens: int | None = None,
    ) -> Any:
        current_tier = self._llama_handle_tier(llm)
        fallback_batch, fallback_ubatch = _llama_loaded_batch_fallback()
        needs_reload, safe_batch, safe_ubatch = llama_prefill_needs_reload(
            model_path=getattr(llm, "_seiso_model_path", model_path) or model_path,
            messages=messages,
            n_ctx=n_ctx,
            loaded_n_batch=int(getattr(llm, "_seiso_n_batch", fallback_batch) or fallback_batch),
            loaded_n_ubatch=int(
                getattr(llm, "_seiso_n_ubatch", fallback_ubatch) or fallback_ubatch
            ),
            loaded_n_gpu_layers=int(getattr(llm, "_seiso_n_gpu_layers", 0) or 0),
            load_tier=current_tier,
            loaded_headroom_mb=getattr(llm, "_seiso_load_headroom_mb", None),
            prompt_tokens=prompt_tokens,
        )
        llm._seiso_last_safe_batch = safe_batch  # noqa: SLF001
        llm._seiso_last_safe_ubatch = safe_ubatch  # noqa: SLF001
        if not needs_reload:
            return llm
        logger.warning(
            "llama.cpp prefill headroom changed before chat — reloading tier=%s "
            "with n_batch=%d n_ubatch=%d",
            current_tier,
            safe_batch,
            safe_ubatch,
        )
        release_cached_memory(sync=True)
        return self._pool.reload_llama(
            model_path,
            n_ctx,
            tier=current_tier,
            batch_override=(safe_batch, safe_ubatch),
            max_tokens=int(getattr(llm, "_seiso_max_tokens", 512) or 512),
        )

    def _llama_prepare_for_generation(
        self,
        llm: Any,
        payload: dict[str, Any],
        *,
        model_path: str,
        messages: list[dict[str, Any]],
        n_ctx: int,
        stream: bool,
    ) -> tuple[Any, dict[str, Any], list[dict[str, Any]], int]:
        """Single fit → prefill guard → decode preflight pass (re-fit only after reload)."""
        before_id = id(llm)
        before_batch = int(getattr(llm, "_seiso_n_batch", 0) or 0)
        before_ubatch = int(getattr(llm, "_seiso_n_ubatch", 0) or 0)
        before_ctx = int(getattr(llm, "_seiso_n_ctx", n_ctx) or n_ctx)
        before_tokens = int(payload.get("max_tokens", 512))

        budget = _fit_llama_messages_to_context(
            llm,
            messages,
            n_ctx=n_ctx,
            max_tokens=before_tokens,
        )
        messages = budget.messages
        if budget.max_tokens != before_tokens:
            payload = dict(payload)
            payload["max_tokens"] = budget.max_tokens
        llm = self._llama_guard_prefill(
            llm,
            model_path=model_path,
            messages=messages,
            n_ctx=n_ctx,
            prompt_tokens=budget.input_tokens,
        )
        llm, payload, messages, n_ctx = self._llama_preflight_decode(
            llm,
            payload,
            model_path=model_path,
            messages=messages,
            n_ctx=n_ctx,
            stream=stream,
            prompt_tokens=budget.input_tokens,
        )
        reloaded = (
            id(llm) != before_id
            or int(getattr(llm, "_seiso_n_batch", 0) or 0) != before_batch
            or int(getattr(llm, "_seiso_n_ubatch", 0) or 0) != before_ubatch
            or int(getattr(llm, "_seiso_n_ctx", n_ctx) or n_ctx) != before_ctx
            or int(payload.get("max_tokens", 512)) != before_tokens
        )
        if reloaded:
            budget = _fit_llama_messages_to_context(
                llm,
                messages,
                n_ctx=n_ctx,
                max_tokens=int(payload.get("max_tokens", 512)),
            )
            messages = budget.messages
            if budget.max_tokens != int(payload.get("max_tokens", 512)):
                payload = dict(payload)
                payload["max_tokens"] = budget.max_tokens
        return llm, payload, messages, n_ctx

    def _llama_preflight_decode(
        self,
        llm: Any,
        payload: dict[str, Any],
        *,
        model_path: str,
        messages: list[dict[str, Any]],
        n_ctx: int,
        stream: bool,
        prompt_tokens: int | None = None,
    ) -> tuple[Any, dict[str, Any], list[dict[str, Any]], int]:
        """Clamp decode shape and reload before tokens when native Linux headroom is tight."""
        try:
            budget = resolve_llama_decode_budget(
                model_path=getattr(llm, "_seiso_model_path", model_path) or model_path,
                free_mb=headroom_mb(),
                n_ctx=n_ctx,
                max_tokens=int(payload.get("max_tokens", 512)),
                n_gpu_layers=int(getattr(llm, "_seiso_n_gpu_layers", 0) or 0),
                load_tier=self._llama_handle_tier(llm),
                prompt_tokens=(
                    prompt_tokens
                    if prompt_tokens is not None
                    else _estimate_prompt_tokens(messages)
                ),
                weights_resident=True,
                stream=stream,
            )
        except Exception:
            logger.warning(
                "llama.cpp decode preflight failed; applying conservative fallback",
                exc_info=True,
            )
            try:
                from seiso.platform import use_linux_nvidia_inference_guards

                native_linux_nvidia = use_linux_nvidia_inference_guards()
            except Exception:
                native_linux_nvidia = False
            if native_linux_nvidia:
                adjusted = dict(payload)
                adjusted["max_tokens"] = max(1, min(int(payload.get("max_tokens", 512)), 256))
                safe_ctx = min(int(n_ctx), 2048)
                messages = trim_llama_messages_to_context(
                    messages,
                    n_ctx=safe_ctx,
                    max_tokens=int(adjusted["max_tokens"]),
                )
                return llm, adjusted, messages, safe_ctx
            return llm, payload, messages, n_ctx

        adjusted = payload
        requested_tokens = int(payload.get("max_tokens", 512))
        if budget.max_tokens < requested_tokens:
            adjusted = dict(payload)
            adjusted["max_tokens"] = budget.max_tokens
            messages = trim_llama_messages_to_context(
                messages,
                n_ctx=budget.n_ctx,
                max_tokens=budget.max_tokens,
            )

        loaded_batch = int(getattr(llm, "_seiso_n_batch", 0) or 0)
        loaded_ubatch = int(getattr(llm, "_seiso_n_ubatch", 0) or 0)
        needs_reload = loaded_batch > budget.n_batch or (
            loaded_ubatch > 0 and loaded_ubatch > budget.n_ubatch
        )
        if not needs_reload:
            llm._seiso_last_safe_batch = budget.n_batch  # noqa: SLF001
            llm._seiso_last_safe_ubatch = budget.n_ubatch  # noqa: SLF001
            return llm, adjusted, messages, n_ctx

        current_tier = self._llama_handle_tier(llm)
        logger.warning(
            "llama.cpp decode headroom tight before %s — reloading tier=%s "
            "with n_batch=%d n_ubatch=%d max_tokens=%d",
            "stream" if stream else "completion",
            current_tier,
            budget.n_batch,
            budget.n_ubatch,
            budget.max_tokens,
        )
        release_cached_memory(sync=True)
        llm = self._pool.reload_llama(
            model_path,
            budget.n_ctx,
            tier=current_tier,
            batch_override=(budget.n_batch, budget.n_ubatch),
            max_tokens=budget.max_tokens,
        )
        actual_ctx = int(getattr(llm, "_seiso_n_ctx", budget.n_ctx) or budget.n_ctx)
        if actual_ctx < n_ctx:
            n_ctx = actual_ctx
            messages = trim_llama_messages_to_context(
                messages,
                n_ctx=actual_ctx,
                max_tokens=budget.max_tokens,
            )
        return llm, adjusted, messages, n_ctx

    def _llama_complete(
        self,
        payload: dict[str, Any],
        model_path: str,
        generation_id: int,
    ) -> str:
        with self._pool.llama_inference_lease():
            return self._llama_complete_locked(payload, model_path, generation_id)

    def _llama_complete_locked(
        self,
        payload: dict[str, Any],
        model_path: str,
        generation_id: int,
    ) -> str:
        messages, n_ctx = _prepare_llama_messages(payload, model_path)
        llm = self._pool.get_llama(
            model_path,
            n_ctx=n_ctx,
            max_tokens=int(payload.get("max_tokens", 512)),
        )
        llm, payload, messages, n_ctx = self._llama_prepare_for_generation(
            llm,
            payload,
            model_path=model_path,
            messages=messages,
            n_ctx=n_ctx,
            stream=False,
        )
        kwargs = llama_completion_kwargs(payload)
        kwargs["stream"] = False
        tools = payload.get("tools_schemas")
        if tools:
            kwargs["tools"] = tools
        recoveries = 0
        while True:
            try:
                out = llm.create_chat_completion(messages=messages, **kwargs)
                break
            except Exception as exc:
                if not is_oom_error(exc):
                    raise
                recoveries += 1
                if recoveries > _MAX_LLAMA_OOM_RECOVERIES:
                    raise RuntimeError(
                        "llama.cpp inference OOM — recovery attempts exhausted"
                    ) from exc
                llm = self._llama_recover_from_oom(
                    llm,
                    model_path=model_path,
                    n_ctx=n_ctx,
                    max_tokens=int(payload.get("max_tokens", 512)),
                )
                actual_ctx = int(getattr(llm, "_seiso_n_ctx", n_ctx) or n_ctx)
                if actual_ctx < int(n_ctx):
                    n_ctx = actual_ctx
                    messages = trim_llama_messages_to_context(
                        messages,
                        n_ctx=actual_ctx,
                        max_tokens=int(payload.get("max_tokens", 512)),
                    )
                llm, payload, messages, n_ctx = self._llama_prepare_for_generation(
                    llm,
                    payload,
                    model_path=model_path,
                    messages=messages,
                    n_ctx=n_ctx,
                    stream=False,
                )
                kwargs = llama_completion_kwargs(payload)
                kwargs["stream"] = False
                if tools:
                    kwargs["tools"] = tools
        if not self._pool.is_generation_active(generation_id):
            return ""
        choices = out.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return message_content_with_tool_calls(message)

    def _llamaswap_payload(self, payload: dict[str, Any], model_path: str) -> dict[str, Any]:
        """Ensure sidecar chat reuses a pinned num_ctx and active keep_alive."""
        out = dict(payload)
        out.setdefault("sidecar_active", True)
        if out.get("sidecar_num_ctx") is not None:
            return out
        # Prefer the preload/pool pin so multi-turn history growth does not
        # re-bucket num_ctx and force an Ollama KV reload.
        pinned = self._pool.pinned_n_ctx(model_path)
        if pinned is not None:
            out["sidecar_num_ctx"] = pinned
            return out
        try:
            from seiso.inference.llamaswap import plan_sidecar_request

            _, planned_ctx, _planned_max = plan_sidecar_request(out, model_path)
            out["sidecar_num_ctx"] = planned_ctx
        except Exception:
            logger.debug("Sidecar request planning failed", exc_info=True)
        return out

    def _llamaswap_complete(
        self,
        payload: dict[str, Any],
        model_path: str,
        generation_id: int,
    ) -> str:
        payload = self._llamaswap_payload(payload, model_path)
        client = self._pool.get_llamaswap(model_path, num_ctx=payload.get("sidecar_num_ctx"))
        engine = getattr(client, "engine", "llamaswap")
        cache_policy = resolve_sidecar_kv_policy(payload, engine=engine)
        prefix_requested = cache_policy.num_keep is not None or cache_policy.cache_prompt is True
        self._last_inference_stats = {
            "backend": engine,
            "cache_mode": "provider-native",
            "prefix_cache": prefix_requested,
            "prefix_cache_mode": (
                "requested"
                if prefix_requested
                else ("provider-managed" if engine == "ollama" else "unconfirmed")
            ),
            "sidecar_num_ctx": payload.get("sidecar_num_ctx"),
        }
        text = client.complete(payload, model_path)
        if not self._pool.is_generation_active(generation_id):
            return ""
        return cast(str, text)

    def _llamaswap_stream(
        self,
        payload: dict[str, Any],
        model_path: str,
        should_stop: Callable[[], bool],
    ) -> Iterator[StreamToken]:
        payload = self._llamaswap_payload(payload, model_path)
        client = self._pool.get_llamaswap(model_path, num_ctx=payload.get("sidecar_num_ctx"))
        engine = getattr(client, "engine", "llamaswap")
        cache_policy = resolve_sidecar_kv_policy(payload, engine=engine)
        prefix_requested = cache_policy.num_keep is not None or cache_policy.cache_prompt is True
        self._last_inference_stats = {
            "backend": engine,
            "cache_mode": "provider-native",
            "prefix_cache": prefix_requested,
            "prefix_cache_mode": (
                "requested"
                if prefix_requested
                else ("provider-managed" if engine == "ollama" else "unconfirmed")
            ),
            "sidecar_num_ctx": payload.get("sidecar_num_ctx"),
            "sidecar_resident_confirmed": False,
        }
        yield from client.stream(
            payload,
            model_path,
            should_stop=should_stop,
            runtime_stats=self._last_inference_stats,
        )

    def _llama_stream(
        self,
        payload: dict[str, Any],
        model_path: str,
        should_stop: Callable[[], bool],
    ) -> Iterator[StreamToken]:
        with self._pool.llama_inference_lease():
            yield from self._llama_stream_locked(payload, model_path, should_stop)

    def _llama_stream_locked(
        self,
        payload: dict[str, Any],
        model_path: str,
        should_stop: Callable[[], bool],
    ) -> Iterator[StreamToken]:
        messages, n_ctx = _prepare_llama_messages(payload, model_path)
        try:
            llm = self._pool.get_llama(
                model_path,
                n_ctx=n_ctx,
                max_tokens=int(payload.get("max_tokens", 512)),
            )
        except ImportError as exc:
            raise RuntimeError("llama-cpp-python not installed") from exc
        llm, payload, messages, n_ctx = self._llama_prepare_for_generation(
            llm,
            payload,
            model_path=model_path,
            messages=messages,
            n_ctx=n_ctx,
            stream=True,
        )

        completion_kwargs = llama_completion_kwargs(payload)
        tools = payload.get("tools_schemas")
        if tools:
            completion_kwargs["tools"] = tools
        tool_buffer = ToolCallDeltaBuffer()
        emitted_text = False
        recoveries = 0
        while True:
            try:
                stream = llm.create_chat_completion(
                    messages=messages,
                    **completion_kwargs,
                )
                for chunk in stream:
                    if should_stop():
                        break
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        emitted_text = True
                        yield StreamToken(content)
                    tool_text = tool_buffer.add(delta.get("tool_calls"))
                    if tool_text:
                        emitted_text = True
                        yield StreamToken(tool_text)
                tool_text = tool_buffer.flush()
                if tool_text and not should_stop():
                    emitted_text = True
                    yield StreamToken(tool_text)
                return
            except Exception as exc:
                if not is_oom_error(exc):
                    raise
                if emitted_text:
                    raise RuntimeError(
                        "llama.cpp inference OOM after streaming began — aborting "
                        "instead of replaying partial output"
                    ) from exc
                recoveries += 1
                if recoveries > _MAX_LLAMA_OOM_RECOVERIES:
                    raise RuntimeError(
                        "llama.cpp inference OOM — recovery attempts exhausted"
                    ) from exc
                tool_buffer = ToolCallDeltaBuffer()
                llm = self._llama_recover_from_oom(
                    llm,
                    model_path=model_path,
                    n_ctx=n_ctx,
                    max_tokens=int(payload.get("max_tokens", 512)),
                )
                actual_ctx = int(getattr(llm, "_seiso_n_ctx", n_ctx) or n_ctx)
                if actual_ctx < int(n_ctx):
                    n_ctx = actual_ctx
                    messages = trim_llama_messages_to_context(
                        messages,
                        n_ctx=actual_ctx,
                        max_tokens=int(payload.get("max_tokens", 512)),
                    )
                llm, payload, messages, n_ctx = self._llama_prepare_for_generation(
                    llm,
                    payload,
                    model_path=model_path,
                    messages=messages,
                    n_ctx=n_ctx,
                    stream=True,
                )
                completion_kwargs = llama_completion_kwargs(payload)
                if tools:
                    completion_kwargs["tools"] = tools


def get_inference_runner() -> LocalInferenceRunner:
    """Process-wide runner — reuses the warmed model pool across requests."""
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                _runner = LocalInferenceRunner()
    return _runner


def reset_inference_runtime(*, wait: bool = True) -> None:
    """Close process-wide inference state for shutdown and test isolation."""
    global _runner, _inference_executor  # pylint: disable=global-statement
    with _runner_lock:
        runner = _runner
        _runner = None
    if runner is not None:
        runner.pool.cancel_and_unload()

    with _inference_executor_lock:
        executor = _inference_executor
        _inference_executor = None
    if executor is not None:
        executor.shutdown(wait=wait, cancel_futures=True)

    from seiso.inference.model_pool import clear_dflash_draft_cache
    from seiso.inference.torch_stream import clear_torch_prefix_cache

    clear_torch_prefix_cache()
    clear_dflash_draft_cache()
    ModelPool.reset_instance(timeout_s=30.0 if wait else 0.0)


async def run_chat(payload: dict[str, Any]) -> str:
    return await get_inference_runner().chat(payload)
