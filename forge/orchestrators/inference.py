"""Inference orchestrator — local, provider, tools, MCP."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Callable

from forge.config import get_settings
from forge.mcp.client import McpManager
from forge.orchestrators.base import Orchestrator
from forge.providers.router import chat_completion
from forge.security.audit import audit_event
from forge.tools.registry import ToolSpec, build_default_registry
from forge.tools.sanitize import wrap_tool_result
from seiso.inference.runner import LocalInferenceRunner
from seiso.models.loader import detect_backend


class InferenceOrchestrator(Orchestrator):
    kind = "inference"

    def __init__(self, sandbox_root) -> None:
        super().__init__(sandbox_root)
        self.mcp = McpManager()
        self._runner = LocalInferenceRunner()

    async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages", [])
        use_tools = payload.get("tools", False)
        allow_code_exec = payload.get("allow_code_exec", False)
        provider = payload.get("provider")
        user_id = payload.get("user_id") or (self.get_job(job_id).user_id if self.get_job(job_id) else None)
        mcp_server_ids: list[str] = payload.get("mcp_server_ids", [])
        settings = get_settings()

        self._emit_log(job_id, f"Messages: {len(messages)}, tools={use_tools}, provider={provider or 'local'}")

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
            if mcp_server_ids:
                for sid in mcp_server_ids:
                    if not self.mcp.get(user_id, sid):
                        raise PermissionError(f"MCP server not connected: {sid}")
            registry = build_default_registry(
                str(self.sandbox_root),
                allow_code_exec=allow_code_exec and settings.allow_code_exec,
                user_id=user_id,
            )
            await self._register_mcp_tools(registry, user_id, mcp_server_ids, on_log)
            reply, _ = await self._tool_loop(payload, messages, registry, on_log, user_id=user_id)
            backend = "local+tools"
        else:
            backend = detect_backend().value
            self._emit_log(job_id, f"Inference backend: {backend}")
            reply = await self._local_chat(payload)

        self._emit_log(job_id, f"Generated {len(reply)} chars")
        return {"content": reply, "backend": backend, "messages": len(messages)}

    async def stream_local(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        """Stream tokens from local inference (no tools/providers)."""
        async for token in self._runner.stream(payload):
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

    async def _register_mcp_tools(
        self,
        registry,
        user_id: str,
        server_ids: list[str],
        on_log: Callable[[str], None],
    ) -> None:
        for sid in server_ids:
            srv = self.mcp.get(user_id, sid)
            if not srv:
                raise PermissionError(f"MCP server not connected: {sid}")
            try:
                for t in await srv.list_tools():
                    name = f"mcp_{sid[:8]}_{t['name']}"
                    tool_name = t["name"]

                    async def mcp_handler(
                        server_id: str = sid,
                        tn: str = tool_name,
                        uid: str = user_id,
                        **kwargs: Any,
                    ) -> str:
                        audit_event("mcp_tool_call", user_id=uid, server_id=server_id, tool=tn)
                        raw = await self.mcp.call(uid, server_id, tn, kwargs)
                        return wrap_tool_result(f"mcp:{server_id}:{tn}", raw)

                    registry.register(
                        ToolSpec(
                            name=name,
                            description=t.get("description", f"MCP tool {tool_name}"),
                            parameters=t.get("inputSchema", {"type": "object", "properties": {}}),
                            handler=lambda **kw: "",
                            async_handler=mcp_handler,
                        )
                    )
            except Exception as exc:
                on_log(f"MCP tool registration failed: {exc}")
                raise

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
