"""Inference orchestrator — local, provider, and tools."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from forge.config import get_settings
from forge.orchestrators.base import Orchestrator
from forge.providers.router import chat_completion
from forge.security.audit import audit_event
from forge.tools.registry import build_default_registry
from seiso.inference.backends import BACKEND_ROUTER
from seiso.inference.runner import get_inference_runner


class InferenceOrchestrator(Orchestrator):
    kind = "inference"

    def __init__(self, sandbox_root: Path) -> None:
        super().__init__(sandbox_root)
        self._runner = get_inference_runner()
        self._active_generation_user_id: str | None = None

    def _generation_owned_by_other(self, user_id: str | None) -> bool:
        return bool(self._active_generation_user_id and self._active_generation_user_id != user_id)

    async def cancel_and_unload_for_user(self, user_id: str | None) -> dict[str, Any]:
        return await self.release_all_inference_memory(user_id)

    async def release_all_inference_memory(self, user_id: str | None) -> dict[str, Any]:
        """Unload local pool and refresh headroom for the next model load."""
        if self._generation_owned_by_other(user_id):
            raise PermissionError("Another user has active inference")
        await self._runner.cancel_and_unload()
        from seiso.memory.protection import release_cached_memory

        release_cached_memory(sync=True)
        from forge.services.hardware import build_vram_status
        from forge.services.inference_models import invalidate_inference_options_cache
        from seiso.hardware.profile import hardware_profile as core_hardware_profile

        core_hardware_profile(force_refresh=True)
        invalidate_inference_options_cache()
        return build_vram_status(self)

    async def cancel_generation_for_user(self, user_id: str | None) -> dict[str, Any]:
        if self._generation_owned_by_other(user_id):
            raise PermissionError("Another user has active inference")
        return await self._runner.cancel_generation()

    async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages", [])
        use_tools = payload.get("tools", False)
        allow_code_exec = payload.get("allow_code_exec", False)
        provider = payload.get("provider")
        job = self.get_job(job_id)
        user_id = payload.get("user_id") or (job.user_id if job else None)
        settings = get_settings()

        self._emit_log(
            job_id, f"Messages: {len(messages)}, tools={use_tools}, provider={provider or 'local'}"
        )

        def on_log(msg: str) -> None:
            self._emit_log(job_id, msg)

        result_router_meta: dict[str, Any] | None = None
        self._active_generation_user_id = str(user_id) if user_id else None
        try:
            if provider:
                reply = await self._provider_chat(provider, payload, messages, user_id=user_id)
                backend = f"provider:{provider.get('provider_type', 'unknown')}"
            elif payload.get("use_model_router"):
                reply, router_meta = await self._router_chat(payload, messages)
                backend = BACKEND_ROUTER
                result_router_meta = router_meta
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
                reply, _ = await self._tool_loop(
                    payload, messages, registry, on_log, user_id=user_id
                )
                backend = "local+tools"
            else:
                active = self._active_backend(payload)
                self._emit_log(job_id, f"Inference backend: {active}")
                reply = await self._local_chat(payload)
                backend = payload.get("inference_backend") or active
        finally:
            if self._active_generation_user_id == (str(user_id) if user_id else None):
                self._active_generation_user_id = None

        self._emit_log(job_id, f"Generated {len(reply)} chars")
        result: dict[str, Any] = {"content": reply, "backend": backend, "messages": len(messages)}
        if result_router_meta:
            result["router"] = result_router_meta
        return result

    async def stream_router(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        """Stream tokens from the Seiso model router."""
        from forge.services.model_router_client import router_stream_chat

        settings = get_settings()
        if not settings.model_router_enabled:
            raise RuntimeError("Model router is not enabled on this server")

        messages = list(payload.get("messages", []))
        async for token in router_stream_chat(
            settings,
            messages,
            model=payload.get("router_model"),
            max_tokens=payload.get("max_tokens", 512),
            temperature=float(payload.get("temperature", 0.7)),
        ):
            yield token

    async def _router_chat(
        self,
        payload: dict[str, Any],
        messages: list[dict],
    ) -> tuple[str, dict[str, Any]]:
        from forge.services.model_router_client import router_chat_completion

        settings = get_settings()
        if not settings.model_router_enabled:
            raise RuntimeError("Model router is not enabled on this server")
        return await router_chat_completion(
            settings,
            messages,
            model=payload.get("router_model"),
            max_tokens=payload.get("max_tokens", 512),
            temperature=float(payload.get("temperature", 0.7)),
        )

    async def stream_local(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        """Stream tokens from local inference (llama.cpp, MLX, or Torch)."""
        async for update in self.stream_local_updates(payload):
            yield update.text

    async def stream_local_updates(self, payload: dict[str, Any]) -> AsyncIterator[Any]:
        """Stream local inference with cumulative output token counts."""
        user_id = payload.get("user_id")
        self._active_generation_user_id = str(user_id) if user_id else None
        try:
            async for update in self._runner.stream_updates(payload):
                yield update
        finally:
            if self._active_generation_user_id == (str(user_id) if user_id else None):
                self._active_generation_user_id = None

    def _active_backend(self, payload: dict[str, Any]) -> str:
        explicit = (payload.get("inference_backend") or "").lower()
        return explicit or "local"

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
            p = {
                **payload,
                "messages": msgs,
                "tools": False,
                "tools_schemas": registry.schemas(),
            }
            return await self._local_chat(p)

        return await run_agent_loop_async(
            generate, messages, registry, on_log=on_log, user_id=user_id
        )

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
        return await self._runner.chat(payload)
