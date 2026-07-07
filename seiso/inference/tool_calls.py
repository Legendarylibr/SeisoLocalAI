"""Helpers for preserving native backend tool calls as parseable chat text."""

from __future__ import annotations

import json
from typing import Any

_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"


def _coerce_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"arguments": raw}
        if isinstance(parsed, dict):
            return parsed
    return {}


def normalize_tool_call(raw: Any) -> dict[str, Any] | None:
    """Convert OpenAI/Ollama tool-call shapes to Seiso's tagged JSON format."""
    if not isinstance(raw, dict):
        return None
    fn = raw.get("function") if isinstance(raw.get("function"), dict) else raw
    name = fn.get("name") if isinstance(fn, dict) else None
    if not isinstance(name, str) or not name:
        return None
    arguments = _coerce_arguments(fn.get("arguments") if isinstance(fn, dict) else None)
    return {"name": name, "arguments": arguments}


def tool_calls_to_text(tool_calls: Any) -> str:
    """Serialize native structured tool calls so the existing parser can see them."""
    if not isinstance(tool_calls, list):
        return ""
    parts: list[str] = []
    for raw in tool_calls:
        call = normalize_tool_call(raw)
        if call is None:
            continue
        payload = json.dumps(call, separators=(",", ":"), ensure_ascii=False)
        parts.append(f"{_TOOL_OPEN}{payload}{_TOOL_CLOSE}")
    return "\n".join(parts)


def message_content_with_tool_calls(message: dict[str, Any]) -> str:
    """Return visible content plus any native tool calls in tagged JSON form."""
    content = str(message.get("content") or "")
    tools = tool_calls_to_text(message.get("tool_calls"))
    if content and tools:
        return f"{content}\n{tools}"
    return content or tools


class ToolCallDeltaBuffer:
    """Accumulate OpenAI-style streaming ``delta.tool_calls`` fragments."""

    def __init__(self) -> None:
        self._calls: dict[int, dict[str, Any]] = {}

    def add(self, tool_calls: Any) -> str:
        if not isinstance(tool_calls, list):
            return ""
        immediate: list[dict[str, Any]] = []
        for offset, raw in enumerate(tool_calls):
            if not isinstance(raw, dict):
                continue
            try:
                index = int(raw.get("index") or offset)
            except (TypeError, ValueError):
                index = offset
            current = self._calls.setdefault(index, {"function": {}})
            function = raw.get("function")
            if isinstance(function, dict):
                target = current.setdefault("function", {})
                name = function.get("name")
                if isinstance(name, str) and name:
                    target["name"] = name
                arguments = function.get("arguments")
                if arguments is not None:
                    if isinstance(arguments, dict) and not target.get("arguments"):
                        target["arguments"] = arguments
                    else:
                        current_arguments = target.get("arguments") or ""
                        if isinstance(current_arguments, dict):
                            current_arguments = json.dumps(
                                current_arguments, separators=(",", ":")
                            )
                        if isinstance(arguments, dict):
                            arguments = json.dumps(arguments, separators=(",", ":"))
                        target["arguments"] = str(current_arguments) + str(arguments)
            elif normalize_tool_call(raw) is not None:
                immediate.append(raw)
        return tool_calls_to_text(immediate)

    def flush(self) -> str:
        calls = [self._calls[index] for index in sorted(self._calls)]
        self._calls.clear()
        return tool_calls_to_text(calls)
