"""Tool registry and self-healing agent loop for inference."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Union

logger = logging.getLogger(__name__)

ToolHandler = Union[Callable[..., Any], Callable[..., Awaitable[Any]]]

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
_MAX_ARTIFACT_BYTES = 512_000

# Legacy pattern kept for stripping assistant text
TOOL_CALL_PATTERN = re.compile(
    rf"{re.escape(TOOL_CALL_OPEN)}\s*(\{{.*?\}})\s*{re.escape(TOOL_CALL_CLOSE)}",
    re.DOTALL,
)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    async_handler: Callable[..., Awaitable[str]] | None = None


@dataclass
class ToolRegistry:
    tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self.tools.values()
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self.tools:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            result = self.tools[name].handler(**arguments)
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as exc:
            logger.warning("Tool %s failed: %s", name, exc)
            return json.dumps({"error": str(exc), "tool": name})

    async def execute_async(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self.tools:
            return json.dumps({"error": f"Unknown tool: {name}"})
        spec = self.tools[name]
        try:
            if spec.async_handler:
                result = await spec.async_handler(**arguments)
            else:
                import asyncio

                result = await asyncio.to_thread(spec.handler, **arguments)
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as exc:
            logger.warning("Tool %s failed: %s", name, exc)
            return json.dumps({"error": str(exc), "tool": name})


def build_default_registry(
    sandbox_root: str | None = None,
    *,
    allow_code_exec: bool = False,
    user_id: str | None = None,
) -> ToolRegistry:
    from forge.tools.code_exec import execute_code
    from forge.tools.web_search import web_search
    from forge.tools.sanitize import wrap_tool_result

    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="web_search",
            description="Search the web for current information. Returns top snippets.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
            handler=lambda query: wrap_tool_result("web_search", web_search(query)),
        )
    )
    if allow_code_exec:
        reg.register(
            ToolSpec(
                name="execute_code",
                description="Run Python code in a sandboxed environment. Returns stdout or error.",
                parameters={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python source to execute"},
                    },
                    "required": ["code"],
                },
                handler=lambda code: wrap_tool_result(
                    "execute_code",
                    execute_code(code, sandbox_root=sandbox_root),
                ),
            )
        )
    reg.register(
        ToolSpec(
            name="write_artifact",
            description="Write a file artifact to the sandbox exports directory.",
            parameters={
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filename", "content"],
            },
            handler=lambda filename, content: wrap_tool_result(
                "write_artifact",
                _write_artifact(filename, content, sandbox_root, user_id),
            ),
        )
    )
    return reg


def _write_artifact(filename: str, content: str, sandbox_root: str | None, user_id: str | None) -> str:
    from pathlib import Path

    from forge.services.user_paths import user_dir
    from seiso.security import resolve_data_dir, safe_join, sanitize_filename

    if len(content.encode()) > _MAX_ARTIFACT_BYTES:
        return json.dumps({"error": f"Content exceeds {_MAX_ARTIFACT_BYTES} bytes"})

    root = Path(sandbox_root) if sandbox_root else resolve_data_dir()
    if user_id:
        base = user_dir(root, user_id, "artifacts")
    else:
        base = root / "artifacts"
    base.mkdir(parents=True, exist_ok=True)
    path = safe_join(base, sanitize_filename(filename))
    path.write_text(content)
    return json.dumps({"path": str(path.name), "bytes": len(content.encode())})


def _extract_json_object(text: str) -> str | None:
    text = text.strip()
    if not text.startswith("{"):
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[: i + 1]
    return None


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    pos = 0
    while True:
        start = text.find(TOOL_CALL_OPEN, pos)
        if start == -1:
            break
        json_start = start + len(TOOL_CALL_OPEN)
        close = text.find(TOOL_CALL_CLOSE, json_start)
        if close == -1:
            break
        blob = text[json_start:close]
        obj_text = _extract_json_object(blob)
        if obj_text:
            try:
                calls.append(json.loads(obj_text))
            except json.JSONDecodeError:
                pass
        pos = close + len(TOOL_CALL_CLOSE)
    return calls


def tools_system_prompt(registry: ToolRegistry) -> str:
    lines = [
        "You have access to tools. To call a tool, emit:",
        '<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>',
        "Available tools:",
    ]
    for t in registry.tools.values():
        lines.append(f"- {t.name}: {t.description}")
    return "\n".join(lines)

