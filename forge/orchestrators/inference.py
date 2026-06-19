"""Inference orchestrator — local, provider, and tools."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from forge.config import get_settings
from forge.orchestrators.base import Orchestrator
from forge.providers.ollama import chat_completion as ollama_chat_completion
from forge.providers.ollama import stream_chat_completion as ollama_stream_chat
from forge.providers.ollama import unload_model as ollama_unload_model
from forge.providers.router import chat_completion
from forge.security.audit import audit_event
from forge.tools.registry import build_default_registry
from seiso.inference.backends import BACKEND_OLLAMA
from seiso.inference.runner import LocalInferenceRunner


class InferenceOrchestrator(Orchestrator):
    kind = "inference"

    def __init__(self, sandbox_root) -> None:
        super().__init__(sandbox_root)
        self._runner = LocalInferenceRunner()
        self._active_ollama_model: str | None = None
        self._active_ollama_base_url = ""
        self._active_generation_user_id: str | None = None

    @property
    def active_ollama_model(self) -> str | None:
        return self._active_ollama_model

    async def release_ollama_model(self) -> None:
        await self._release_ollama_model()

    def _generation_owned_by_other(self, user_id: str | None) -> bool:
        return bool(self._active_generation_user_id and self._active_generation_user_id != user_id)

    async def cancel_and_unload_for_user(self, user_id: str | None) -> dict[str, Any]:
        if self._generation_owned_by_other(user_id):
            raise PermissionError("Another user has active inference")
        return await self._runner.cancel_and_unload()

    async def cancel_generation_for_user(self, user_id: str | None) -> dict[str, Any]:
        if self._generation_owned_by_other(user_id):
            raise PermissionError("Another user has active inference")
        return await self._runner.cancel_generation()

    async def prepare_ollama_model(self, model: str, base_url: str = "") -> None:
        await self._runner.cancel_and_unload()
        await self._release_ollama_model(next_model=model, next_base_url=base_url)
        self._active_ollama_model = model
        self._active_ollama_base_url = base_url

    async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages", [])
        use_tools = payload.get("tools", False)
        allow_code_exec = payload.get("allow_code_exec", False)
        provider = payload.get("provider")
        job = self.get_job(job_id)
        user_id = payload.get("user_id") or (job.user_id if job else None)
        settings = get_settings()

        self._emit_log(job_id, f"Messages: {len(messages)}, tools={use_tools}, provider={provider or 'local'}")

        def on_log(msg: str) -> None:
            self._emit_log(job_id, msg)

        self._active_generation_user_id = str(user_id) if user_id else None
        try:
            if provider:
                reply = await self._provider_chat(provider, payload, messages, user_id=user_id)
                backend = f"provider:{provider.get('provider_type', 'unknown')}"
            elif use_tools:
                if not user_id:
                    raise PermissionError("user_id required for tool execution")
                if not settings.allow_tools:
                    raise PermissionError("Tools are disabled on this server")
                if allow_code_exec and not settings.allow_code_exec:
                    raise PermissionError("Code execution is disabled on this server")
                registry = build_default_registry(
                    str(self.sandbox_root),
                    allow_code_exec=allow_code_exec and settings.allow_code_exec,
                    user_id=user_id,
                )
                reply, _ = await self._tool_loop(payload, messages, registry, on_log, user_id=user_id)
                backend = "local+tools"
            else:
                active = self._active_backend(payload)
                self._emit_log(job_id, f"Inference backend: {active}")
                if active == BACKEND_OLLAMA:
                    reply = await self._ollama_chat(payload, messages)
                    backend = BACKEND_OLLAMA
                else:
                    reply = await self._local_chat(payload)
                    backend = payload.get("inference_backend") or active
        finally:
            if self._active_generation_user_id == (str(user_id) if user_id else None):
                self._active_generation_user_id = None

        self._emit_log(job_id, f"Generated {len(reply)} chars")
        return {"content": reply, "backend": backend, "messages": len(messages)}

    async def stream_local(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        """Stream tokens from local inference (llama.cpp, MLX, Torch, or Ollama)."""
        user_id = payload.get("user_id")
        self._active_generation_user_id = str(user_id) if user_id else None
        try:
            if self._active_backend(payload) == BACKEND_OLLAMA:
                async for token in self._ollama_stream(payload):
                    yield token
                return
            await self.release_ollama_model()
            async for token in self._runner.stream(payload):
                yield token
        finally:
            if self._active_generation_user_id == (str(user_id) if user_id else None):
                self._active_generation_user_id = None

    def _active_backend(self, payload: dict[str, Any]) -> str:
        explicit = (payload.get("inference_backend") or "").lower()
        if explicit == BACKEND_OLLAMA or payload.get("ollama_model"):
            return BACKEND_OLLAMA
        return explicit or "local"

    def _ollama_model_name(self, payload: dict[str, Any]) -> str:
        model = payload.get("ollama_model")
        if model:
            return model
        model_path = payload.get("model_path")
        if model_path:
            return Path(model_path).stem
        raise ValueError("ollama_model required for Ollama inference")

    async def _ollama_chat(self, payload: dict[str, Any], messages: list[dict]) -> str:
        model = await self._prepare_ollama_switch(payload)
        return await ollama_chat_completion(
            messages,
            model=model,
            max_tokens=payload.get("max_tokens", 512),
            base_url=payload.get("ollama_base_url", ""),
        )

    async def _ollama_stream(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        model = await self._prepare_ollama_switch(payload)
        async for token in ollama_stream_chat(
            payload.get("messages", []),
            model=model,
            max_tokens=payload.get("max_tokens", 512),
            base_url=payload.get("ollama_base_url", ""),
        ):
            yield token

    async def _prepare_ollama_switch(self, payload: dict[str, Any]) -> str:
        model = self._ollama_model_name(payload)
        base_url = payload.get("ollama_base_url", "")
        await self.prepare_ollama_model(model, base_url)
        return model

    async def _release_ollama_model(
        self,
        *,
        next_model: str | None = None,
        next_base_url: str = "",
    ) -> None:
        if not self._active_ollama_model:
            return
        if (
            next_model
            and self._active_ollama_model == next_model
            and self._active_ollama_base_url == next_base_url
        ):
            return
        model = self._active_ollama_model
        base_url = self._active_ollama_base_url
        self._active_ollama_model = None
        self._active_ollama_base_url = ""
        await ollama_unload_model(model, base_url)

    async def _tool_loop(
        self,
        payload: dict,
        messages: list[dict],
        registry,
        on_log: Callable[[str], None],
        *,
        user_id: str,
    ) -> tuple[str, list[dict]]:
        from forge.tools.agent_loop import run_agent_loop_async

        async def generate(msgs: list[dict]) -> str:
            p = {**payload, "messages": msgs, "tools": False}
            return await self._local_chat(p)

        return await run_agent_loop_async(generate, messages, registry, on_log=on_log, user_id=user_id)

    async def _provider_chat(
        self,
        provider: dict,
        payload: dict,
        messages: list[dict],
        *,
        user_id: str | None,
    ) -> str:
        audit_event(
            "provider_chat",
            user_id=user_id,
            provider_type=provider.get("provider_type"),
            base_url=provider.get("config", {}).get("base_url", ""),
        )
        return await chat_completion(
            provider["provider_type"],
            provider.get("config", {}),
            messages,
            max_tokens=payload.get("max_tokens", 512),
        )

    async def _local_chat(self, payload: dict[str, Any]) -> str:
        await self.release_ollama_model()
        return await self._runner.chat(payload)
