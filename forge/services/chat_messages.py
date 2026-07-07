"""Trusted chat message assembly — never trust client-supplied tool/system history."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import HTTPException

from forge.db.store import Database
from forge.services.model_prompts import (
    chat_system_prompt,
    model_switch_system_prompt,
    resolve_model_key,
)
from forge.tools.sanitize import normalize_text

_UNTRUSTED_ROLES = frozenset({"tool", "function", "system", "developer"})
_DEFAULT_CONTEXT_CHAR_BUDGET = 24_000
# Native Linux NVIDIA multi-turn chats OOM more easily as history grows;
# keep a tighter default so prefill stays within safe VRAM.
_NATIVE_LINUX_CONTEXT_CHAR_BUDGET = 12_000
_DEFAULT_DECAY_HALF_LIFE_SECONDS = 3600.0
_DEFAULT_DECAY_MIN_WEIGHT = 0.05
_OMISSION_MARKER = "[...older content omitted...]\n"


def _default_context_char_budget() -> int:
    try:
        from seiso.platform import use_linux_nvidia_inference_guards

        if use_linux_nvidia_inference_guards():
            return _NATIVE_LINUX_CONTEXT_CHAR_BUDGET
    except Exception:
        pass
    return _DEFAULT_CONTEXT_CHAR_BUDGET


def _context_char_budget() -> int:
    raw = os.environ.get("SEISO_CHAT_CONTEXT_CHARS", "").strip()
    if not raw:
        return _default_context_char_budget()
    else:
        try:
            return max(1_000, int(raw))
        except ValueError:
            return _default_context_char_budget()


def _decay_half_life_seconds() -> float:
    raw = os.environ.get("SEISO_CHAT_DECAY_HALF_LIFE_SECONDS", "").strip()
    if not raw:
        return _DEFAULT_DECAY_HALF_LIFE_SECONDS
    try:
        return max(60.0, float(raw))
    except ValueError:
        return _DEFAULT_DECAY_HALF_LIFE_SECONDS


def _decay_min_weight() -> float:
    raw = os.environ.get("SEISO_CHAT_DECAY_MIN_WEIGHT", "").strip()
    if not raw:
        return _DEFAULT_DECAY_MIN_WEIGHT
    try:
        return max(0.01, min(1.0, float(raw)))
    except ValueError:
        return _DEFAULT_DECAY_MIN_WEIGHT


def _parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def _message_age_seconds(message: dict[str, str], *, anchor: datetime) -> float:
    ts = _parse_timestamp(message.get("created_at"))
    if ts is None:
        return _decay_half_life_seconds()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return max(0.0, (anchor - ts).total_seconds())


def _decay_weight(age_seconds: float, half_life: float) -> float:
    if age_seconds <= 0:
        return 1.0
    weight = 0.5 ** (age_seconds / half_life)
    return max(_decay_min_weight(), weight)


def _message_cost(message: dict[str, str]) -> int:
    return len(message.get("role", "")) + len(message.get("content", "")) + 16


def _trim_message_content(message: dict[str, str], budget: int) -> dict[str, str]:
    content = message.get("content", "")
    overhead = len(message.get("role", "")) + 16
    content_budget = max(1, budget - overhead)
    if len(content) <= content_budget:
        return message
    if content_budget <= len(_OMISSION_MARKER):
        return {**message, "content": content[-content_budget:]}
    keep = content_budget - len(_OMISSION_MARKER)
    return {**message, "content": f"{_OMISSION_MARKER}{content[-keep:]}"}


def trim_messages_to_context(
    messages: list[dict[str, str]],
    *,
    budget: int | None = None,
) -> list[dict[str, str]]:
    """Keep the newest turns within a bounded prompt budget with time-based decay."""
    if not messages:
        return []

    limit = budget or _context_char_budget()
    half_life = _decay_half_life_seconds()
    anchor = datetime.now(timezone.utc)

    latest = _trim_message_content(messages[-1], limit)
    trimmed: list[dict[str, str]] = [latest]
    used = _message_cost(latest)

    for msg in reversed(messages[:-1]):
        remaining = limit - used
        if remaining <= 64:
            break

        age = _message_age_seconds(msg, anchor=anchor)
        weight = _decay_weight(age, half_life)
        msg_budget = max(64, int(remaining * weight))
        adjusted = _trim_message_content(msg, min(msg_budget, remaining))
        cost = _message_cost(adjusted)
        if cost <= remaining:
            trimmed.insert(0, adjusted)
            used += cost

    return trimmed


def _strip_message_metadata(message: dict[str, str]) -> dict[str, str]:
    return {"role": message["role"], "content": message["content"]}


def prepare_chat_context(
    history: list[dict[str, str]],
    *,
    model_key: str,
    tools_enabled: bool,
    prior_model_key: str | None = None,
    knowledge_context: str | None = None,
) -> list[dict[str, str]]:
    """Apply decay trimming, model system prompt, and mid-thread model-switch bridge."""
    trimmed = trim_messages_to_context(history)
    out: list[dict[str, str]] = []

    system = chat_system_prompt(model_key, tools_enabled=tools_enabled)
    if system:
        out.append({"role": "system", "content": system})

    if knowledge_context:
        out.append({"role": "system", "content": knowledge_context})

    if (
        prior_model_key
        and prior_model_key != model_key
        and trimmed
        and any(m.get("role") == "assistant" for m in trimmed)
    ):
        out.append(
            {
                "role": "system",
                "content": model_switch_system_prompt(prior_model_key, model_key),
            }
        )

    out.extend(_strip_message_metadata(m) for m in trimmed)
    return out


async def build_trusted_messages(
    db: Database,
    *,
    thread_id: str | None,
    client_messages: list[dict],
    persist_user: bool = True,
    user_id: str | None = None,
    model_id: str | None = None,
    model_path: str | None = None,
    tools_enabled: bool = False,
    knowledge_context: str | None = None,
) -> tuple[list[dict[str, str]], str | None]:
    """Return (messages, new_user_content).

    Loads canonical user/assistant history from the DB when thread_id is set.
    Client may only supply the latest user turn; tool/system roles are rejected.
    """
    if not client_messages:
        raise HTTPException(400, "messages required")

    for msg in client_messages:
        role = str(msg.get("role", "")).lower()
        if role in _UNTRUSTED_ROLES:
            raise HTTPException(400, f"Untrusted message role: {role}")
        if role == "assistant" and thread_id:
            raise HTTPException(400, "Assistant history must come from the server")

    last = client_messages[-1]
    if last.get("role") != "user":
        raise HTTPException(400, "Last message must be from user")
    content = normalize_text(str(last.get("content", ""))).strip()
    if not content:
        raise HTTPException(400, "Empty message")

    model_key = resolve_model_key(
        model_id=model_id,
        model_path=model_path,
    )
    track_model = model_id or (model_key if model_key != "default" else None)

    if thread_id:
        thread, stored = await db.get_thread_with_messages(thread_id, user_id or "")
        prior_model_key: str | None = None
        if thread:
            prior_model_key = thread.get("model_id") or None

        history: list[dict[str, str]] = [
            {
                "role": m["role"],
                "content": m["content"],
                "created_at": m.get("created_at", ""),
            }
            for m in stored
            if m.get("role") in ("user", "assistant")
        ]
        model_changed = bool(
            track_model and thread and (thread.get("model_id") or None) != track_model
        )
        need_persist = (
            not history or history[-1]["role"] != "user" or history[-1]["content"] != content
        )
        if need_persist:
            if persist_user:
                await db.add_message(
                    thread_id,
                    "user",
                    content,
                    model_id=track_model if model_changed else None,
                )
            history.append({"role": "user", "content": content, "created_at": ""})
        elif model_changed:
            await db.update_thread_model(thread_id, track_model)

        return (
            prepare_chat_context(
                history,
                model_key=model_key,
                tools_enabled=tools_enabled,
                prior_model_key=prior_model_key,
                knowledge_context=knowledge_context,
            ),
            content,
        )

    if len(client_messages) != 1:
        raise HTTPException(400, "New chats accept a single user message")
    return (
        prepare_chat_context(
            [{"role": "user", "content": content}],
            model_key=model_key,
            tools_enabled=tools_enabled,
            knowledge_context=knowledge_context,
        ),
        content,
    )
