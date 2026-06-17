"""Trusted chat message assembly — never trust client-supplied tool/system history."""

from __future__ import annotations

from fastapi import HTTPException

from forge.db.store import Database

_UNTRUSTED_ROLES = frozenset({"tool", "function", "system"})


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
        return history, content

    if len(client_messages) != 1:
        raise HTTPException(400, "New chats accept a single user message")
    return [{"role": "user", "content": content}], content
