"""Nemotron-Orchestrator-8B route selection (ToolOrchestra-style tool calling).

Uses ``nvidia/Nemotron-Orchestrator-8B`` to pick a specialist route via the
``answer`` tool, matching the evaluation harness in NVlabs/ToolOrchestra.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from seiso.model_router.catalog import SpecialistCatalog, SpecialistRoute

logger = logging.getLogger(__name__)

TOOL_ANSWER = "answer"


@dataclass(frozen=True)
class OrchestratorSelection:
    route: SpecialistRoute
    orchestrator_alias: str
    reasoning: str
    raw_tool_calls: list[dict[str, Any]]


def orchestrator_alias_for(route: SpecialistRoute) -> str:
    if route.orchestrator_alias:
        return route.orchestrator_alias
    return f"seiso-{route.route_id}-1"


def alias_route_map(catalog: SpecialistCatalog) -> dict[str, SpecialistRoute]:
    return {orchestrator_alias_for(route): route for route in catalog}


def build_answer_tools(catalog: SpecialistCatalog) -> list[dict[str, Any]]:
    """Build OpenAI-style tool definitions for specialist delegation."""
    aliases: list[str] = []
    detail_lines: list[str] = []
    for route in catalog:
        alias = orchestrator_alias_for(route)
        aliases.append(alias)
        desc = route.orchestrator_detail()
        detail_lines.append(f"{alias}: {desc}")

    model_description = (
        "The specialist model used to answer. Choices: "
        + json.dumps(aliases)
        + ". "
        + " ".join(detail_lines)
    )
    return [
        {
            "type": "function",
            "function": {
                "name": TOOL_ANSWER,
                "description": (
                    "Delegate the problem to a specialist model and return its answer."
                ),
                "parameters": {
                    "type": "object",
                    "title": "parameters",
                    "properties": {
                        "model": {
                            "description": model_description,
                            "type": "string",
                            "enum": aliases,
                        }
                    },
                    "required": ["model"],
                },
            },
        }
    ]


def problem_from_messages(messages: list[dict[str, Any]]) -> str:
    """Flatten chat history into a single problem string for the orchestrator."""
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip()
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [
                str(p.get("text", ""))
                for p in content
                if isinstance(p, dict) and p.get("type", "text") == "text"
            ]
            content = "\n".join(text_parts)
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "user":
            parts.append(content.strip())
        elif role == "assistant":
            parts.append(f"Assistant: {content.strip()}")
        elif role == "system":
            parts.append(f"System: {content.strip()}")
    return "\n\n".join(parts)


def parse_tool_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool calls from an OpenAI-compatible chat completion response."""
    choices = payload.get("choices") or []
    if not choices:
        return []
    message = choices[0].get("message") or {}
    raw_calls = message.get("tool_calls") or []
    parsed: list[dict[str, Any]] = []
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        name = fn.get("name") or call.get("name")
        arguments = fn.get("arguments") or call.get("arguments") or "{}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        parsed.append({"name": name, "arguments": arguments})
    return parsed


def selection_from_tool_calls(
    catalog: SpecialistCatalog,
    tool_calls: list[dict[str, Any]],
    *,
    fallback_route_id: str,
) -> OrchestratorSelection | None:
    """Map Nemotron tool calls to a specialist route."""
    mapping = alias_route_map(catalog)
    for call in tool_calls:
        if call.get("name") != TOOL_ANSWER:
            continue
        args = call.get("arguments") or {}
        alias = str(args.get("model", "")).strip()
        route = mapping.get(alias)
        if route is None:
            logger.warning("orchestrator picked unknown alias %r", alias)
            continue
        return OrchestratorSelection(
            route=route,
            orchestrator_alias=alias,
            reasoning=f"nemotron_tool=answer model={alias}",
            raw_tool_calls=tool_calls,
        )

    try:
        route = catalog.by_id(fallback_route_id)
    except KeyError:
        route = next(iter(catalog))
    return OrchestratorSelection(
        route=route,
        orchestrator_alias="",
        reasoning="nemotron_fallback_no_valid_tool_call",
        raw_tool_calls=tool_calls,
    )


async def select_route_via_nemotron(
    client: httpx.AsyncClient,
    *,
    orchestrator_url: str,
    orchestrator_model: str,
    catalog: SpecialistCatalog,
    messages: list[dict[str, Any]],
    fallback_route_id: str,
    timeout: float = 120.0,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> OrchestratorSelection:
    """Ask Nemotron-Orchestrator-8B which specialist should handle the request."""
    base = orchestrator_url.rstrip("/")
    problem = problem_from_messages(messages)
    tools = build_answer_tools(catalog)
    chat = [
        {"role": "system", "content": "You are good at using tools."},
        {
            "role": "user",
            "content": f"Problem: {problem}\n\nChoose an appropriate tool.",
        },
    ]
    payload = {
        "model": orchestrator_model,
        "messages": chat,
        "tools": tools,
        "tool_choice": "required",
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    resp = await client.post(
        f"{base}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"orchestrator HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    tool_calls = parse_tool_calls(data)
    if not tool_calls:
        logger.warning("orchestrator returned no tool calls")

    selection = selection_from_tool_calls(catalog, tool_calls, fallback_route_id=fallback_route_id)
    return selection
