"""Inference orchestrator — local, provider, and tools."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from forge.config import get_settings
from forge.orchestrators.base import Orchestrator
from forge.providers.ollama import chat_completion as ollama_chat_completion
from forge.providers.ollama import stream_chat_completion as ollama_stream_chat
from forge.providers.router import chat_completion
from forge.security.audit import audit_event
from forge.security.autodefense import (
    DefenseBlockedError,
    defense_enabled,
    scan_messages,
    scan_output,
)
from forge.tools.registry import build_default_registry
from seiso.inference.backends import BACKEND_OLLAMA
from seiso.inference.runner import LocalInferenceRunner


class InferenceOrchestrator(Orchestrator):
    kind = "inference"

    def __init__(self, sandbox_root) -> None:
        super().__init__(sandbox_root)
        self._runner = LocalInferenceRunner()

    async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages", [])
        use_tools = payload.get("tools", False)
        allow_code_exec = payload.get("allow_code_exec", False)
        provider = payload.get("provider")
        user_id = payload.get("user_id") or (self.get_job(job_id).user_id if self.get_job(job_id) else None)
        use_defense = defense_enabled(settings := get_settings(), request_flag=payload.get("defense"))
        session_id = payload.get("thread_id") or job_id
        defense_meta: dict[str, Any] | None = None

        self._emit_log(job_id, f"Messages: {len(messages)}, tools={use_tools}, provider={provider or 'local'}")

        if use_defense:
            self._emit_log(job_id, "AutoDefense: scanning input")
            messages, input_result = await scan_messages(
                messages,
                session_id=session_id,
                user_id=user_id,
                settings=settings,
            )
            payload = {**payload, "messages": messages}
            if input_result and not input_result.unavailable:
                defense_meta = input_result.to_dict()
                self._emit_log(
                    job_id,
                    f"AutoDefense input: action={input_result.action} risk={input_result.risk_score}",
                )

        def on_log(msg: str) -> None:
            self._emit_log(job_id, msg)

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

        if use_defense:
            self._emit_log(job_id, "AutoDefense: scanning output")
            reply, output_result = await scan_output(
                messages,
                reply,
                session_id=session_id,
                user_id=user_id,
                settings=settings,
            )
            if not output_result.unavailable:
                defense_meta = output_result.to_dict()
                self._emit_log(
                    job_id,
                    f"AutoDefense output: action={output_result.action} risk={output_result.risk_score}",
                )

        self._emit_log(job_id, f"Generated {len(reply)} chars")
        result: dict[str, Any] = {"content": reply, "backend": backend, "messages": len(messages)}
        if defense_meta:
            result["defense"] = defense_meta
        return result

    async def stream_local(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        """Stream tokens from local inference (llama.cpp, MLX, Torch, or Ollama)."""
        settings = get_settings()
        use_defense = defense_enabled(settings, request_flag=payload.get("defense"))
        messages = list(payload.get("messages", []))

        if use_defense:
            messages, _ = await scan_messages(
                messages,
                session_id=payload.get("thread_id"),
                user_id=payload.get("user_id"),
                settings=settings,
            )
            payload = {**payload, "messages": messages}

        if self._active_backend(payload) == BACKEND_OLLAMA:
            async for token in self._ollama_stream(payload):
                yield token
            return
        async for token in self._runner.stream(payload):
            yield token

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
        return await ollama_chat_completion(
            messages,
            model=self._ollama_model_name(payload),
            max_tokens=payload.get("max_tokens", 512),
            base_url=payload.get("ollama_base_url", ""),
        )

    async def _ollama_stream(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        async for token in ollama_stream_chat(
            payload.get("messages", []),
            model=self._ollama_model_name(payload),
            max_tokens=payload.get("max_tokens", 512),
            base_url=payload.get("ollama_base_url", ""),
        ):
            yield token

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
        return await self._runner.chat(payload)

    async def _run(self, job_id: str, payload: dict[str, Any]) -> None:
        """Preserve AutoDefense metadata when a scan blocks the interaction."""
        import asyncio

        from forge.orchestrators.base import JobStatus

        rec = self._jobs[job_id]
        try:
            result = await self.execute(job_id, payload)
            rec.result = result
            rec.status = JobStatus.COMPLETED
        except DefenseBlockedError as exc:
            rec.status = JobStatus.FAILED
            rec.error = str(exc)
            if exc.result:
                rec.result = {"defense": exc.result.to_dict()}
            self._emit_log(job_id, f"ERROR: {exc}")
        except asyncio.CancelledError:
            rec.status = JobStatus.CANCELLED
            self._emit_log(job_id, "Job cancelled")
            raise
        except Exception as exc:
            rec.status = JobStatus.FAILED
            rec.error = str(exc)
            self._emit_log(job_id, f"ERROR: {exc}")
        finally:
            self._subprocesses.pop(job_id, None)
            self._finish_logs(job_id)
