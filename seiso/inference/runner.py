"""Local inference runner — VRAM-managed via ModelPool."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from queue import Empty
from typing import Any

from seiso.env import env_int
from seiso.inference.backends import (
    BACKEND_LLAMASWAP,
    BACKEND_MLX,
    BACKEND_TORCH,
    is_dflash_draft,
    prepare_model_path,
    resolve_local_backend,
)
from seiso.inference.model_pool import get_dflash_draft, get_model_pool
from seiso.inference.speculative import (
    DFlashDraftSpeculativeBundle,
    default_num_speculative_tokens,
    iter_speculative_tokens,
    iter_speculative_tokens_dflash,
)
from seiso.inference.streaming import StreamToken, StreamUpdate
from seiso.inference.tuning import (
    configure_torch_inference,
    estimate_llama_n_ctx,
    extract_mlx_token_text,
    generate_with_cache_fallback,
    llama_completion_kwargs,
    mlx_stream_kwargs,
    torch_generate_kwargs,
)
from seiso.memory.protection import (
    LlamaLoadTier,
    is_oom_error,
    llama_next_recovery_tier,
    llama_prefill_needs_reload,
    release_cached_memory,
    sanitize_inference_payload,
)
from seiso.models.chat_format import format_messages_for_prompt

logger = logging.getLogger(__name__)

_STREAM_DONE = object()
_MAX_LLAMA_OOM_RECOVERIES = 3
_runner: LocalInferenceRunner | None = None
_runner_lock = threading.Lock()


def _stream_batch_chars() -> int:
    """Chars to batch after the first token; lower = snappier UI, higher = fewer SSE events."""
    return max(1, env_int("SEISO_STREAM_BATCH_CHARS", 16))


def _torch_stream_timeout_s() -> int:
    """Poll interval for detecting failed Torch generation threads."""
    return max(1, env_int("SEISO_TORCH_STREAM_TIMEOUT_S", 2))


class _StreamError:
    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


class LocalInferenceRunner:
    """Runs chat against local MLX, PyTorch, or llama.cpp with VRAM management."""

    def __init__(self) -> None:
        self._pool = get_model_pool()

    @property
    def pool(self):
        """Active model pool (public accessor for Forge services)."""
        return self._pool

    def resolve_route(
        self, payload: dict[str, Any], model_path: str
    ) -> tuple[str, str]:
        return self._resolve_route(payload, model_path)

    def warm_model(self, payload: dict[str, Any]) -> None:
        """Load a model into the pool without generating (preload / ping)."""
        model_path = payload["model_path"]
        route, resolved_path = self._resolve_route(payload, model_path)
        if route == "mlx":
            self._pool.get_mlx(resolved_path)
        elif route == "torch":
            self._pool.get_torch(resolved_path)
        elif route == "llamaswap":
            self._pool.get_llamaswap(resolved_path)
        elif route == "speculative":
            draft_path = payload.get("draft_model_path")
            if not draft_path:
                raise ValueError("draft_model_path required for speculative preload")
            if is_dflash_draft(draft_path):
                self._pool.get_torch(resolved_path, load_in_4bit=True)
                get_dflash_draft(
                    draft_path, n_ctx=self._estimate_dflash_n_ctx(payload, draft_path)
                )
            else:
                self._pool.get_torch_speculative(
                    resolved_path, draft_path, load_in_4bit=True
                )
        else:
            from seiso.inference.tuning import estimate_llama_n_ctx

            messages = payload.get("messages") or []
            n_ctx = payload.get("n_ctx") or estimate_llama_n_ctx(
                messages,
                max_tokens=int(payload.get("max_tokens", 1)),
                model_path=resolved_path,
                model_format=payload.get("model_format"),
            )
            self._pool.get_llama(resolved_path, n_ctx=n_ctx)

    async def chat(self, payload: dict[str, Any]) -> str:
        loop = asyncio.get_running_loop()
        if payload.get("tools_schemas"):
            payload = sanitize_inference_payload(payload)
            model_path = payload.get("model_path") or payload.get("model_id")
            if not model_path:
                raise ValueError("model_path or model_id required")
            route, resolved_path = self._resolve_route(payload, model_path)
            if route not in {"llama", "llamaswap"}:
                raise ValueError(
                    "Tool calling is only supported with GGUF local backends"
                )
            generation_id = self._pool.bump_generation()
            await self._ensure_model_switch(resolved_path, route=route)
            if route == "llamaswap":
                return await loop.run_in_executor(
                    None,
                    lambda: self._llamaswap_complete(
                        payload, resolved_path, generation_id
                    ),
                )
            return await loop.run_in_executor(
                None,
                lambda: self._llama_complete(payload, resolved_path, generation_id),
            )

        payload = sanitize_inference_payload(payload)
        model_path = payload.get("model_path") or payload.get("model_id")
        if not model_path:
            raise ValueError("model_path or model_id required")

        route, resolved_path = self._resolve_route(payload, model_path)
        generation_id = self._pool.bump_generation()
        await self._ensure_model_switch(
            resolved_path, draft_path=payload.get("draft_model_path"), route=route
        )
        return await loop.run_in_executor(
            None,
            lambda: self._complete(payload, resolved_path, route, generation_id),
        )

    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        async for update in self.stream_updates(payload):
            yield update.text

    async def stream_updates(
        self, payload: dict[str, Any]
    ) -> AsyncIterator[StreamUpdate]:
        payload = sanitize_inference_payload(payload)
        model_path = payload.get("model_path") or payload.get("model_id")
        if not model_path:
            raise ValueError("model_path or model_id required")

        route, resolved_path = self._resolve_route(payload, model_path)
        draft_path = payload.get("draft_model_path")
        generation_id = self._pool.bump_generation()
        await self._ensure_model_switch(
            resolved_path, draft_path=draft_path, route=route
        )

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[StreamUpdate | object] = asyncio.Queue()

        def should_stop() -> bool:
            return not self._pool.is_generation_active(generation_id)

        def producer() -> None:
            buffer: list[str] = []
            buffered = 0
            output_tokens = 0
            flushed_once = False
            batch_chars = _stream_batch_chars()
            try:
                for part in self._iter_tokens(
                    payload, resolved_path, route, should_stop
                ):
                    if should_stop():
                        break
                    output_tokens += part.new_tokens
                    buffer.append(part.text)
                    buffered += len(part.text)
                    if not flushed_once:
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            StreamUpdate(
                                text="".join(buffer), output_tokens=output_tokens
                            ),
                        )
                        buffer.clear()
                        buffered = 0
                        flushed_once = True
                    elif buffered >= batch_chars:
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            StreamUpdate(
                                text="".join(buffer), output_tokens=output_tokens
                            ),
                        )
                        buffer.clear()
                        buffered = 0
                if buffer and not should_stop():
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        StreamUpdate(text="".join(buffer), output_tokens=output_tokens),
                    )
            except Exception as exc:
                if buffer and not should_stop():
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        StreamUpdate(text="".join(buffer), output_tokens=output_tokens),
                    )
                if not should_stop():
                    loop.call_soon_threadsafe(queue.put_nowait, _StreamError(exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _STREAM_DONE)

        threading.Thread(target=producer, daemon=True).start()

        while True:
            if should_stop():
                break
            item = await queue.get()
            if item is _STREAM_DONE:
                break
            if isinstance(item, _StreamError):
                raise item.exc
            yield item

    async def unload(self) -> dict:
        loop = asyncio.get_running_loop()

        await loop.run_in_executor(None, self._pool.cancel_and_unload)
        return self._pool.status()

    async def cancel_and_unload(self) -> dict:
        return await self.unload()

    async def cancel_generation(self) -> dict:
        """Stop active streams without unloading the warmed model."""
        self._pool.bump_generation()
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
                and self._pool.normalize_path(active_path)
                == self._pool.normalize_path(model_path)
                and self._pool.normalize_path(active_draft)
                == self._pool.normalize_path(draft_path)
            ):
                return
            if not is_dflash_draft(draft_path):
                # Torch+Torch speculative bundles are distinct pool handles; unload
                # a warmed single target before loading target+draft together.
                await loop.run_in_executor(None, lambda: self._pool.prepare_for_load())
                return
            # prepare target (torch for verification in dflash case too)
            await loop.run_in_executor(
                None, lambda: self._pool.prepare_for_load(model_path, BACKEND_TORCH)
            )
            return

        if active_draft:
            await loop.run_in_executor(
                None, lambda: self._pool.prepare_for_load(model_path)
            )
            return

        backend = BACKEND_LLAMASWAP if route == "llamaswap" else None
        if self._pool.would_switch_model(model_path, backend):
            await loop.run_in_executor(
                None, lambda: self._pool.prepare_for_load(model_path, backend)
            )

    def _resolve_route(
        self, payload: dict[str, Any], model_path: str
    ) -> tuple[str, str]:
        if payload.get("draft_model_path"):
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
        return int(
            payload.get("n_ctx")
            or estimate_llama_n_ctx(
                payload.get("messages") or [],
                max_tokens=int(payload.get("max_tokens", 512)),
                model_path=draft_path,
                model_format="gguf",
            )
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
        if route == "speculative":
            chunks: list[str] = []

            def should_stop() -> bool:
                return not self._pool.is_generation_active(generation_id)

            for token in self._torch_speculative_stream(
                payload, model_path, should_stop
            ):
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

        configure_torch_inference()

        if is_dflash_draft(draft_path):
            # dFlash speculative: fast GGUF dflash draft (llama.cpp) + target verifier (torch)
            target_model, target_tok = self._pool.get_torch(
                model_path, load_in_4bit=True
            )
            draft_llm = get_dflash_draft(
                draft_path, n_ctx=self._estimate_dflash_n_ctx(payload, draft_path)
            )

            bundle = DFlashDraftSpeculativeBundle(
                target_model=target_model,
                target_tokenizer=target_tok,
                draft_llm=draft_llm,
                draft_tokenizer=target_tok,  # vocab alignment expected for dflash
            )

            messages = payload.get("messages", [])
            prompt = format_messages_for_prompt(messages, target_tok)
            temperature = float(payload.get("temperature", 0.0))
            max_new_tokens = int(payload.get("max_tokens", 512))
            num_speculative_tokens = default_num_speculative_tokens(payload)

            yield from iter_speculative_tokens_dflash(
                bundle=bundle,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                num_speculative_tokens=num_speculative_tokens,
                temperature=temperature,
                should_stop=should_stop,
            )
            return

        # Original torch + torch speculative
        bundle = self._pool.get_torch_speculative(
            model_path, draft_path, load_in_4bit=True
        )
        messages = payload.get("messages", [])
        prompt = format_messages_for_prompt(messages, bundle.target_tokenizer)
        temperature = float(payload.get("temperature", 0.0))
        max_new_tokens = int(payload.get("max_tokens", 512))
        num_speculative_tokens = default_num_speculative_tokens(payload)

        try:
            yield from iter_speculative_tokens(
                bundle=bundle,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                num_speculative_tokens=num_speculative_tokens,
                temperature=temperature,
                should_stop=should_stop,
            )
        except Exception as exc:
            if not is_oom_error(exc):
                raise
            release_cached_memory(sync=True)
            reduced = max(32, max_new_tokens // 2)
            logger.warning(
                "Speculative inference OOM — retrying with max_new_tokens=%s", reduced
            )
            yield from iter_speculative_tokens(
                bundle=bundle,
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

    def _torch_complete(
        self,
        payload: dict[str, Any],
        model_path: str,
        generation_id: int,
    ) -> str:
        import torch

        configure_torch_inference()
        model, tokenizer = self._pool.get_torch(model_path, load_in_4bit=True)
        prompt = format_messages_for_prompt(payload.get("messages", []), tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt")
        device = self._torch_input_device(model)
        inputs = {
            k: v.to(device, non_blocking=getattr(device, "type", "") == "cuda")
            for k, v in inputs.items()
        }
        input_len = int(inputs["input_ids"].shape[-1])
        gen_kwargs = torch_generate_kwargs(
            payload,
            inputs,
            streamer=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_kwargs.pop("streamer", None)

        with torch.inference_mode():
            try:
                generated = generate_with_cache_fallback(model, gen_kwargs)
            except Exception as exc:
                if not is_oom_error(exc):
                    raise
                release_cached_memory(sync=True)
                reduced = dict(gen_kwargs)
                reduced["max_new_tokens"] = max(
                    32, int(reduced.get("max_new_tokens", 512)) // 2
                )
                logger.warning(
                    "Torch inference OOM — retrying with max_new_tokens=%s",
                    reduced["max_new_tokens"],
                )
                generated = generate_with_cache_fallback(model, reduced)

        if not self._pool.is_generation_active(generation_id):
            return ""
        output_ids = generated[0][input_len:]
        return str(tokenizer.decode(output_ids, skip_special_tokens=True))

    def _torch_stream(
        self,
        payload: dict[str, Any],
        model_path: str,
        should_stop: Callable[[], bool],
    ) -> Iterator[StreamToken]:
        import torch
        from transformers import TextIteratorStreamer

        configure_torch_inference()
        model, tokenizer = self._pool.get_torch(model_path, load_in_4bit=True)
        messages = payload.get("messages", [])
        prompt = format_messages_for_prompt(messages, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt")
        device = self._torch_input_device(model)
        inputs = {
            k: v.to(device, non_blocking=getattr(device, "type", "") == "cuda")
            for k, v in inputs.items()
        }

        streamer = TextIteratorStreamer(
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
            with torch.inference_mode():
                try:
                    generate_with_cache_fallback(model, gen_kwargs)
                except Exception as exc:
                    if not is_oom_error(exc):
                        generation_errors.append(exc)
                        with contextlib.suppress(Exception):
                            streamer.on_finalized_text("", stream_end=True)
                        return
                    release_cached_memory(sync=True)
                    reduced = dict(gen_kwargs)
                    reduced["max_new_tokens"] = max(
                        32, int(reduced.get("max_new_tokens", 512)) // 2
                    )
                    logger.warning(
                        "Torch inference OOM — retrying with max_new_tokens=%s",
                        reduced["max_new_tokens"],
                    )
                    try:
                        generate_with_cache_fallback(model, reduced)
                    except Exception as retry_exc:
                        generation_errors.append(retry_exc)
                        with contextlib.suppress(Exception):
                            streamer.on_finalized_text("", stream_end=True)

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()
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
        thread.join(timeout=0)
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
    ) -> Any:
        current = self._llama_handle_tier(llm)
        next_tier = llama_next_recovery_tier(current)
        if next_tier is None:
            raise RuntimeError("llama.cpp inference OOM — recovery tiers exhausted")
        logger.warning(
            "llama.cpp inference OOM at tier=%s — reloading at tier=%s",
            current,
            next_tier,
        )
        release_cached_memory(sync=True)
        return self._pool.reload_llama(model_path, n_ctx, tier=next_tier)

    def _llama_guard_prefill(
        self,
        llm: Any,
        *,
        model_path: str,
        messages: list[dict[str, Any]],
        n_ctx: int,
    ) -> Any:
        current_tier = self._llama_handle_tier(llm)
        needs_reload, safe_batch, safe_ubatch = llama_prefill_needs_reload(
            model_path=getattr(llm, "_seiso_model_path", model_path) or model_path,
            messages=messages,
            n_ctx=n_ctx,
            loaded_n_batch=int(getattr(llm, "_seiso_n_batch", 4096) or 4096),
            loaded_n_gpu_layers=int(getattr(llm, "_seiso_n_gpu_layers", 0) or 0),
            load_tier=current_tier,
            loaded_headroom_mb=getattr(llm, "_seiso_load_headroom_mb", None),
        )
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
        )

    def _llama_complete(
        self,
        payload: dict[str, Any],
        model_path: str,
        generation_id: int,
    ) -> str:
        messages = payload.get("messages", [])
        n_ctx = payload.get("n_ctx") or estimate_llama_n_ctx(
            messages,
            max_tokens=int(payload.get("max_tokens", 512)),
            model_path=model_path,
            model_format=payload.get("model_format"),
        )
        llm = self._pool.get_llama(model_path, n_ctx=n_ctx)
        llm = self._llama_guard_prefill(
            llm, model_path=model_path, messages=messages, n_ctx=n_ctx
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
                    llm, model_path=model_path, n_ctx=n_ctx
                )
        if not self._pool.is_generation_active(generation_id):
            return ""
        message = out["choices"][0].get("message") or {}
        return str(message.get("content") or "")

    def _llamaswap_complete(
        self,
        payload: dict[str, Any],
        model_path: str,
        generation_id: int,
    ) -> str:
        client = self._pool.get_llamaswap(model_path)
        text = client.complete(payload, model_path)
        if not self._pool.is_generation_active(generation_id):
            return ""
        return text

    def _llamaswap_stream(
        self,
        payload: dict[str, Any],
        model_path: str,
        should_stop: Callable[[], bool],
    ) -> Iterator[StreamToken]:
        client = self._pool.get_llamaswap(model_path)
        yield from client.stream(payload, model_path, should_stop=should_stop)

    def _llama_stream(
        self,
        payload: dict[str, Any],
        model_path: str,
        should_stop: Callable[[], bool],
    ) -> Iterator[StreamToken]:
        messages = payload.get("messages", [])
        n_ctx = payload.get("n_ctx") or estimate_llama_n_ctx(
            messages,
            max_tokens=int(payload.get("max_tokens", 512)),
            model_path=model_path,
            model_format=payload.get("model_format"),
        )
        try:
            llm = self._pool.get_llama(model_path, n_ctx=n_ctx)
        except ImportError as exc:
            raise RuntimeError("llama-cpp-python not installed") from exc
        llm = self._llama_guard_prefill(
            llm, model_path=model_path, messages=messages, n_ctx=n_ctx
        )

        completion_kwargs = llama_completion_kwargs(payload)
        emitted_text = False
        recoveries = 0
        # #region agent log
        from seiso.agent_debug_log import agent_debug_enabled, agent_debug_log

        if agent_debug_enabled():
            agent_debug_log(
                hypothesis_id="C",
                location="runner.py:_llama_stream:before_prefill",
                message="starting llama.cpp chat prefill",
                data={
                    "model": Path(model_path).name,
                    "n_ctx": n_ctx,
                    "max_tokens": completion_kwargs.get("max_tokens"),
                    "load_tier": getattr(llm, "_seiso_load_tier", None),
                    "n_gpu_layers": getattr(llm, "_seiso_n_gpu_layers", None),
                    "message_count": len(messages),
                    "prompt_chars": sum(
                        len(str(m.get("content") or "")) for m in messages
                    ),
                },
            )
        # #endregion
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
                llm = self._llama_recover_from_oom(
                    llm, model_path=model_path, n_ctx=n_ctx
                )


def get_inference_runner() -> LocalInferenceRunner:
    """Process-wide runner — reuses the warmed model pool across requests."""
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                _runner = LocalInferenceRunner()
    return _runner


async def run_chat(payload: dict[str, Any]) -> str:
    return await get_inference_runner().chat(payload)
