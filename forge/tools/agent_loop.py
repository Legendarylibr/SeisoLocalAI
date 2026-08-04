"""Async agent loop for inference orchestrator."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from forge.security.audit import audit_event, hash_audit_payload
from forge.tools.registry import (
    TOOL_CALL_PATTERN,
    ToolRegistry,
    parse_tool_calls,
    tools_system_prompt,
)

_MAX_TOOL_CALLS_PER_ROUND = 8


async def run_agent_loop_async(
    generate_fn: Callable[[list[dict]], Awaitable[str]],
    messages: list[dict],
    registry: ToolRegistry,
    *,
    max_rounds: int = 5,
    on_log: Callable[[str], None] | None = None,
    user_id: str | None = None,
    model_key: str | None = None,
) -> tuple[str, list[dict]]:
    history = list(messages)
    system = tools_system_prompt(registry, model_key=model_key)
    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": system})
    elif "tool_call" not in history[0].get("content", ""):
        history[0]["content"] = history[0]["content"] + "\n\n" + system

    for round_i in range(max_rounds):
        reply = await generate_fn(history)
        calls = parse_tool_calls(reply, model_key=model_key)

        if not calls:
            from forge.services.llm_output import strip_spurious_chat_artifacts

            clean = strip_spurious_chat_artifacts(
                TOOL_CALL_PATTERN.sub("", reply).strip()
            )
            return clean or reply, history + [
                {"role": "assistant", "content": clean or reply}
            ]

        if len(calls) > _MAX_TOOL_CALLS_PER_ROUND:
            dropped = calls[_MAX_TOOL_CALLS_PER_ROUND:]
            calls = calls[:_MAX_TOOL_CALLS_PER_ROUND]
            audit_event(
                "tool_calls_capped",
                user_id=user_id,
                round=round_i + 1,
                kept=len(calls),
                dropped=len(dropped),
                dropped_tools=[str(c.get("name") or "") for c in dropped],
            )
            if on_log:
                on_log(
                    f"Tool round {round_i + 1}: capped to {len(calls)} "
                    f"(dropped {len(dropped)})"
                )

        if on_log:
            on_log(f"Tool round {round_i + 1}: {len(calls)} call(s)")

        history.append({"role": "assistant", "content": reply})
        for call in calls:
            name = call.get("name", "")
            args = call.get("arguments", {})
            if not isinstance(args, dict):
                args = {}
            if on_log:
                on_log(f"  → {name}({json.dumps(args)[:120]})")
            audit_event(
                "tool_call",
                user_id=user_id,
                tool=name,
                round=round_i + 1,
                args_sha256=hash_audit_payload(args),
                registered=name in registry.tools,
            )
            result = await registry.execute_async(name, args)
            audit_event(
                "tool_result",
                user_id=user_id,
                tool=name,
                round=round_i + 1,
                result_sha256=hash_audit_payload(result),
                result_chars=len(result) if isinstance(result, str) else None,
            )
            history.append({"role": "tool", "name": name, "content": result})

    final = await generate_fn(history)
    return final, history + [{"role": "assistant", "content": final}]
