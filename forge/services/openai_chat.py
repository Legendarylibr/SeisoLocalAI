"""OpenAI-compatible chat payload preparation."""

from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException

from forge.api.schemas.openai import ChatCompletionRequest, ChatMessage
from forge.config import ForgeSettings
from forge.db.store import Database
from forge.services.inference_chat import prepare_local_chat_target
from forge.services.user_paths import is_local_filesystem_path
from forge.tools.sanitize import normalize_text

_UNTRUSTED_OPENAI_ROLES = frozenset({"tool", "function", "system", "developer"})
_UNVERIFIED_ASSISTANT_PREFIX = "[UNVERIFIED_PRIOR_ASSISTANT]\n"


def normalize_openai_messages(body: ChatCompletionRequest) -> list[dict[str, str]]:
    """Reject privileged roles; downgrade client assistant turns to unverified user data."""
    if not body.messages:
        raise HTTPException(400, "At least one user message is required")
    if body.messages[-1].role.lower() != "user":
        raise HTTPException(400, "Last message must be from user")

    messages: list[dict[str, str]] = []
    for m in body.messages:
        role = m.role.lower()
        if role in _UNTRUSTED_OPENAI_ROLES:
            raise HTTPException(400, f"Untrusted message role: {m.role}")
        content = normalize_text(m.content if isinstance(m.content, str) else json.dumps(m.content))
        if role == "assistant":
            messages.append(
                {
                    "role": "user",
                    "content": f"{_UNVERIFIED_ASSISTANT_PREFIX}{content}",
                }
            )
            continue
        if role != "user":
            raise HTTPException(400, f"Unsupported message role: {m.role}")
        messages.append({"role": "user", "content": content})
    if not messages:
        raise HTTPException(400, "At least one user message is required")
    if messages[-1]["role"] != "user":
        raise HTTPException(400, "Last message must be from user")
    return messages


def estimate_token_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped.split()))


def prompt_token_estimate(messages: list[ChatMessage]) -> int:
    return sum(
        estimate_token_count(m.content if isinstance(m.content, str) else json.dumps(m.content))
        for m in messages
    )


async def prepare_openai_chat_payload(
    body: ChatCompletionRequest,
    user_id: str,
    db: Database,
    settings: ForgeSettings,
) -> dict[str, Any]:
    """Resolve and sanitize via the shared local-chat path."""
    messages = normalize_openai_messages(body)
    max_tokens = body.max_tokens or 512

    if body.model in ("default", "seiso"):
        from forge.services.inference_models import list_inference_options

        options = await list_inference_options(db, user_id, hardware_aware=False)
        selected = next(
            (
                o
                for o in options
                if o.get("selectable", True)
                and (o.get("format") or "").lower() == "gguf"
                and o.get("kind") == "local"
            ),
            None,
        )
        if selected is None:
            selected = next(
                (o for o in options if o.get("selectable", True) and o.get("kind") == "local"),
                None,
            )
        if selected is None:
            raise HTTPException(400, "No local model available — download from Hub")
        target = await prepare_local_chat_target(
            db,
            user_id,
            settings,
            model_id=selected["id"],
            inference_backend="auto",
            max_tokens=max_tokens,
            messages=messages,
            check_memory=True,
            sanitize=True,
        )
    elif is_local_filesystem_path(body.model):
        target = await prepare_local_chat_target(
            db,
            user_id,
            settings,
            model_path=body.model,
            inference_backend="auto",
            max_tokens=max_tokens,
            messages=messages,
            check_memory=True,
            sanitize=True,
        )
    else:
        match = await db.get_model(body.model, user_id)
        if match is None:
            match = await db.get_model_by_name(user_id, body.model)
        if not match:
            raise HTTPException(404, f"Model not found in inventory: {body.model}")
        target = await prepare_local_chat_target(
            db,
            user_id,
            settings,
            model_id=match["id"],
            inference_backend="auto",
            max_tokens=max_tokens,
            messages=messages,
            check_memory=True,
            sanitize=True,
        )

    payload: dict[str, Any] = {
        "model_path": target.get("model_path"),
        "messages": messages,
        "max_tokens": target.get("max_tokens", max_tokens),
        "temperature": body.temperature,
        "tools": bool(body.tools),
        "inference_backend": target.get("inference_backend", "auto"),
    }
    if target.get("model_format"):
        payload["model_format"] = target["model_format"]
    if target.get("model_metadata"):
        payload["model_metadata"] = target["model_metadata"]
    if target.get("n_ctx") is not None:
        payload["n_ctx"] = target["n_ctx"]
    return payload
