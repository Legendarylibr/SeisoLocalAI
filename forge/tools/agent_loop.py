"""Async agent loop for inference orchestrator."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from forge.security.audit import audit_event
from forge.tools.registry import ToolRegistry, parse_tool_calls, tools_system_prompt


async def run_agent_loop_async(
    generate_fn: Callable[[list[dict]], Awaitable[str]],
    messages: list[dict],
    registry: ToolRegistry,
    *,
    max_rounds: int = 5,
    on_log: Callable[[str], None] | None = None,
    user_id: str | None = None,
) -> tuple[str, list[dict]]:
    history = list(messages)
    system = tools_system_prompt(registry)
    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": system})
    elif "tool_call" not in history[0].get("content", ""):
        history[0]["content"] = history[0]["content"] + "\n\n" + system

    for round_i in range(max_rounds):
        reply = await generate_fn(history)
        calls = parse_tool_calls(reply)

        if not calls:
            from forge.tools.registry import TOOL_CALL_PATTERN

            clean = TOOL_CALL_PATTERN.sub("", reply).strip()
            return clean or reply, history + [{"role": "assistant", "content": clean or reply}]

        if on_log:
            on_log(f"Tool round {round_i + 1}: {len(calls)} call(s)")

        history.append({"role": "assistant", "content": reply})
        for call in calls:
            name = call.get("name", "")
            args = call.get("arguments", {})
            if on_log:
                on_log(f"  → {name}({json.dumps(args)[:120]})")
            audit_event("tool_call", user_id=user_id, tool=name)
            result = await registry.execute_async(name, args)
            history.append({"role": "tool", "name": name, "content": result})

    final = await generate_fn(history)
    return final, history + [{"role": "assistant", "content": final}]
