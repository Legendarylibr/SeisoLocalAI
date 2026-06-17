"""Local inference runner — VRAM-managed via ModelPool."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from seiso.inference.backends import (
    BACKEND_MLX,
    BACKEND_TORCH,
    prepare_model_path,
    resolve_local_backend,
)
from seiso.inference.model_pool import get_model_pool
from seiso.models.chat_format import format_messages_for_prompt

logger = logging.getLogger(__name__)

_STREAM_DONE = object()


class _StreamError:
    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


class LocalInferenceRunner:
    """Runs chat against local MLX, PyTorch, or llama.cpp with VRAM management."""

    def __init__(self) -> None:
        self._pool = get_model_pool()

    async def chat(self, payload: dict[str, Any]) -> str:
        chunks: list[str] = []
        async for token in self.stream(payload):
            chunks.append(token)
        return "".join(chunks)

    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        model_path = payload.get("model_path") or payload.get("model_id")
        if not model_path:
            raise ValueError("model_path or model_id required")

        route, resolved_path = self._resolve_route(payload, model_path)
        generation_id = self._pool.bump_generation()
        await self._ensure_model_switch(resolved_path)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | object] = asyncio.Queue()

        def should_stop() -> bool:
            return not self._pool.is_generation_active(generation_id)

        def producer() -> None:
            try:
                for token in self._iter_tokens(payload, resolved_path, route, should_stop):
                    if should_stop():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            except Exception as exc:
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
            yield str(item)

    async def unload(self) -> dict:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._pool.cancel_and_unload)
        return self._pool.status()

    async def cancel_and_unload(self) -> dict:
        return await self.unload()

    async def _ensure_model_switch(self, model_path: str) -> None:
        status = self._pool.status()
        active_path = status.get("path")
        if active_path and self._pool.normalize_path(active_path) != self._pool.normalize_path(model_path):
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._pool.unload_all)

    def _resolve_route(self, payload: dict[str, Any], model_path: str) -> tuple[str, str]:
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
        return "llama", resolved

    def _iter_tokens(
        self,
        payload: dict[str, Any],
        model_path: str,
        route: str,
        should_stop: Callable[[], bool],
    ) -> Iterator[str]:
        if route == "mlx":
            yield from self._mlx_stream(payload, model_path, should_stop)
        elif route == "torch":
            yield from self._torch_stream(payload, model_path, should_stop)
        else:
            yield from self._llama_stream(payload, model_path, should_stop)

    def _mlx_stream(
        self,
        payload: dict[str, Any],
        model_path: str,
        should_stop: Callable[[], bool],
    ) -> Iterator[str]:
        try:
            from mlx_lm import generate
        except ImportError as exc:
            raise RuntimeError("MLX not available — install mlx-lm on macOS") from exc

        model, tokenizer = self._pool.get_mlx(model_path)
        prompt = format_messages_for_prompt(payload.get("messages", []), tokenizer)
        max_tokens = payload.get("max_tokens", 512)

        try:
            from mlx_lm import stream_generate

            for token in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
                if should_stop():
                    break
                if isinstance(token, tuple):
                    yield token[0]
                else:
                    yield str(token)
            return
        except (ImportError, TypeError):
            pass

        if not should_stop():
            yield generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)

    def _torch_stream(
        self,
        payload: dict[str, Any],
        model_path: str,
        should_stop: Callable[[], bool],
    ) -> Iterator[str]:
        import torch
        from transformers import TextIteratorStreamer

        model, tokenizer = self._pool.get_torch(model_path, load_in_4bit=True)
        messages = payload.get("messages", [])
        prompt = format_messages_for_prompt(messages, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            **inputs,
            "max_new_tokens": payload.get("max_tokens", 512),
            "streamer": streamer,
            "do_sample": payload.get("temperature", 0) > 0,
            "temperature": max(payload.get("temperature", 0.7), 0.01),
        }
        if payload.get("temperature", 0) <= 0:
            gen_kwargs["do_sample"] = False
            gen_kwargs.pop("temperature", None)

        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()
        for text in streamer:
            if should_stop():
                break
            if text:
                yield text
        thread.join(timeout=0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _llama_stream(
        self,
        payload: dict[str, Any],
        model_path: str,
        should_stop: Callable[[], bool],
    ) -> Iterator[str]:
        try:
            llm = self._pool.get_llama(model_path, n_ctx=payload.get("n_ctx", 4096))
        except ImportError as exc:
            raise RuntimeError("llama-cpp-python not installed") from exc

        messages = payload.get("messages", [])
        stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=payload.get("max_tokens", 512),
            stream=True,
        )
        for chunk in stream:
            if should_stop():
                break
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content


async def run_chat(payload: dict[str, Any]) -> str:
    runner = LocalInferenceRunner()
    return await runner.chat(payload)
