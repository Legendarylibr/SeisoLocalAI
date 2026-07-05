"""Trusted message assembly and session persistence for Seiso Code agent."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from forge.config import ForgeSettings
from forge.security.code_policy import normalize_user_text
from seiso.security import safe_join

_UNTRUSTED_ROLES = frozenset({"tool", "system", "developer"})
_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_MAX_SESSION_MESSAGES = 80


def validate_session_id(session_id: str | None) -> str | None:
    if session_id is None:
        return None
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(
            400,
            "session_id must be 1–64 alphanumeric characters, hyphens, or underscores",
        )
    return session_id


def _session_path(settings: ForgeSettings, user_id: str, session_id: str) -> Path:
    base = safe_join(settings.data_dir, "code_agent_sessions", user_id)
    base.mkdir(parents=True, exist_ok=True)
    return safe_join(base, f"{session_id}.json")


def load_session_history(
    settings: ForgeSettings, user_id: str, session_id: str
) -> list[dict[str, str]]:
    path = _session_path(settings, user_id, session_id)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    history: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).lower()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            history.append({"role": role, "content": content})
    return history[-_MAX_SESSION_MESSAGES:]


def persist_session_history(
    settings: ForgeSettings,
    user_id: str,
    session_id: str,
    history: list[dict[str, str]],
) -> None:
    path = _session_path(settings, user_id, session_id)
    trimmed = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("role") in {"user", "assistant"} and m.get("content")
    ][-_MAX_SESSION_MESSAGES:]
    path.write_text(json.dumps(trimmed, indent=0), encoding="utf-8")
    path.chmod(0o600)


def build_trusted_code_agent_messages(
    settings: ForgeSettings,
    *,
    user_id: str,
    session_id: str | None,
    client_messages: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], str | None, str | None]:
    """Return (messages, latest_user_content, session_id).

    Client may only supply the latest user turn when session_id is set.
    Assistant/system/tool roles from the client are rejected.
    """
    session_id = validate_session_id(session_id)

    if not client_messages:
        raise HTTPException(400, "messages are required")

    for msg in client_messages:
        role = str(msg.get("role", "")).lower()
        if role in _UNTRUSTED_ROLES:
            raise HTTPException(400, f"Untrusted message role: {role}")
        if role == "assistant":
            raise HTTPException(400, "Assistant history must come from the server")

    last = client_messages[-1]
    if last.get("role") != "user":
        raise HTTPException(400, "Last message must be from user")
    content = str(last.get("content", "")).strip()
    if not content:
        raise HTTPException(400, "Empty message")
    content = normalize_user_text(content)
    if not content:
        raise HTTPException(400, "Empty message after sanitization")

    if session_id:
        if len(client_messages) != 1:
            raise HTTPException(
                400, "Ongoing agent sessions accept a single user message"
            )
        history = load_session_history(settings, user_id, session_id)
        if history and history[-1]["role"] == "user" and history[-1]["content"] == content:
            return history, content, session_id
        history.append({"role": "user", "content": content})
        return history, content, session_id

    if len(client_messages) != 1:
        raise HTTPException(400, "New agent chats accept a single user message")
    return [{"role": "user", "content": content}], content, session_id


def append_assistant_turn(
    settings: ForgeSettings,
    user_id: str,
    session_id: str,
    history: list[dict[str, str]],
    assistant_content: str,
) -> None:
    updated = list(history)
    updated.append({"role": "assistant", "content": assistant_content})
    persist_session_history(settings, user_id, session_id, updated)
