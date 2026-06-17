"""Local inference runner — VRAM-managed via ModelPool."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from seiso.inference.backends import (
    BACKEND_LLAMACPP,
    BACKEND_MLX,
    BACKEND_TORCH,
    _is_gguf_path,
    prepare_model_path,
    resolve_gguf_file,
    resolve_local_backend,
)
from seiso.inference.model_pool import get_model_pool
from seiso.models.chat_format import format_messages_for_prompt


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
        await self._ensure_model_switch(resolved_path)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def producer() -> None:
            try:
                for token in self._iter_tokens(payload, resolved_path, route):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=producer, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    async def unload(self) -> dict:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._pool.unload_all)
        return self._pool.status()

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

    def _iter_tokens(self, payload: dict[str, Any], model_path: str, route: str) -> Iterator[str]:
        if route == "mlx":
            yield from self._mlx_stream(payload, model_path)
        elif route == "torch":
            yield from self._torch_stream(payload, model_path)
        else:
            yield from self._llama_stream(payload, model_path)

    def _mlx_stream(self, payload: dict[str, Any], model_path: str) -> Iterator[str]:
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
                if isinstance(token, tuple):
                    yield token[0]
                else:
                    yield str(token)
            return
        except (ImportError, TypeError):
            pass

        yield generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)

    def _torch_stream(self, payload: dict[str, Any], model_path: str) -> Iterator[str]:
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
            if text:
                yield text
        thread.join(timeout=0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _llama_stream(self, payload: dict[str, Any], model_path: str) -> Iterator[str]:
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
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content


async def run_chat(payload: dict[str, Any]) -> str:
    runner = LocalInferenceRunner()
    return await runner.chat(payload)
