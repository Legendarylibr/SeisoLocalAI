"""Trusted chat message assembly — never trust client-supplied tool/system history."""

from __future__ import annotations

import os

from fastapi import HTTPException

from forge.db.store import Database

_UNTRUSTED_ROLES = frozenset({"tool", "function", "system", "developer"})
_DEFAULT_CONTEXT_CHAR_BUDGET = 24_000
_OMISSION_MARKER = "[...older content omitted...]\n"


def _context_char_budget() -> int:
    raw = os.environ.get("SEISO_CHAT_CONTEXT_CHARS", "").strip()
    if not raw:
        base = _DEFAULT_CONTEXT_CHAR_BUDGET
    else:
        try:
            base = max(1_000, int(raw))
        except ValueError:
            base = _DEFAULT_CONTEXT_CHAR_BUDGET
    try:
        from seiso.memory.protection import headroom_mb

        headroom = headroom_mb()
        if headroom < 4096:
            return min(base, 8_000)
        if headroom < 8192:
            return min(base, 16_000)
    except Exception:
        pass
    return base


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
    """Keep the newest turns within a bounded prompt budget."""
    if not messages:
        return []

    limit = budget or _context_char_budget()
    latest = _trim_message_content(messages[-1], limit)
    trimmed: list[dict[str, str]] = [latest]
    used = _message_cost(latest)

    for msg in reversed(messages[:-1]):
        cost = _message_cost(msg)
        if used + cost <= limit:
            trimmed.insert(0, msg)
            used += cost

    return trimmed


async def build_trusted_messages(
    db: Database,
    *,
    thread_id: str | None,
    client_messages: list[dict],
    persist_user: bool = True,
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
    content = str(last.get("content", "")).strip()
    if not content:
        raise HTTPException(400, "Empty message")

    if thread_id:
        stored = await db.get_messages(thread_id)
        history: list[dict[str, str]] = [
            {"role": m["role"], "content": m["content"]}
            for m in stored
            if m.get("role") in ("user", "assistant")
        ]
        if not history or history[-1]["role"] != "user" or history[-1]["content"] != content:
            if persist_user:
                await db.add_message(thread_id, "user", content)
            history.append({"role": "user", "content": content})
        return trim_messages_to_context(history), content

    if len(client_messages) != 1:
        raise HTTPException(400, "New chats accept a single user message")
    return trim_messages_to_context([{"role": "user", "content": content}]), content
