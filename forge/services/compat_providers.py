"""Resolve multi-GPU / provider targets for the Compat API without breaking local models.

External agents (Cursor, Continue, custom clients) call ``/v1/models`` and
``/v1/chat/completions``. Provider models are **additive** — local inventory
and ``default`` / ``seiso`` behavior is unchanged.
"""

from __future__ import annotations

import json
import time
from typing import Any

from forge.db.store import Database
from forge.providers.router import allowed_chat_provider_types, is_chat_provider_type


def provider_model_id(provider_id: str) -> str:
    return f"provider:{provider_id}"


def parse_provider_model_id(model: str) -> str | None:
    raw = (model or "").strip()
    if raw.startswith("provider:"):
        pid = raw.split(":", 1)[1].strip()
        return pid or None
    return None


async def list_compat_provider_models(db: Database, user_id: str) -> list[dict[str, Any]]:
    """OpenAI-style model entries for configured chat providers (local + cloud multi-GPU)."""
    allowed = allowed_chat_provider_types()
    rows = await db.list_providers(user_id)
    created = int(time.time())
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        ptype = row["provider_type"].lower()
        if ptype not in allowed:
            continue
        try:
            config = json.loads(row["config_json"])
        except (TypeError, json.JSONDecodeError):
            config = {}
        pid = row["id"]
        entry_id = provider_model_id(pid)
        if entry_id not in seen_ids:
            seen_ids.add(entry_id)
            out.append(
                {
                    "id": entry_id,
                    "object": "model",
                    "created": created,
                    "owned_by": f"seiso-{ptype}",
                    "root": row.get("name") or entry_id,
                    "permission": [],
                }
            )
        # Friendly alias: upstream model string, only if it does not collide later.
        alias = str(config.get("compat_model_id") or config.get("model") or "").strip()
        if alias and alias not in seen_ids and alias not in {"default", "seiso"}:
            seen_ids.add(alias)
            out.append(
                {
                    "id": alias,
                    "object": "model",
                    "created": created,
                    "owned_by": f"seiso-{ptype}",
                    "root": entry_id,
                    "permission": [],
                }
            )
    return out


async def resolve_compat_provider(
    db: Database,
    user_id: str,
    model: str,
    *,
    local_model_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """Return provider payload for Compat routing, or None to use local inference.

    Resolution order (non-breaking):
    1. Explicit ``provider:<id>``
    2. Alias match on provider config.model / name **only when** that string is
       not already a local inventory model id/name
    """
    allowed = allowed_chat_provider_types()
    rows = await db.list_providers(user_id)
    by_id = {r["id"]: r for r in rows if r["provider_type"].lower() in allowed}

    explicit = parse_provider_model_id(model)
    if explicit:
        row = by_id.get(explicit)
        if not row:
            return None
        return _provider_payload(row)

    # Do not steal local inventory ids.
    if local_model_ids and model in local_model_ids:
        return None
    if model in {"default", "seiso"}:
        return None

    model_l = model.strip().lower()
    for row in by_id.values():
        try:
            config = json.loads(row["config_json"])
        except (TypeError, json.JSONDecodeError):
            config = {}
        candidates = {
            str(config.get("model") or "").strip().lower(),
            str(config.get("compat_model_id") or "").strip().lower(),
            str(row.get("name") or "").strip().lower(),
        }
        if model_l and model_l in candidates:
            return _provider_payload(row)
    return None


def _provider_payload(row: dict[str, Any]) -> dict[str, Any]:
    ptype = row["provider_type"].lower()
    if not is_chat_provider_type(ptype):
        raise ValueError(f"Provider type not enabled: {ptype}")
    try:
        config = json.loads(row["config_json"])
    except (TypeError, json.JSONDecodeError):
        config = {}
    return {
        "provider_id": row["id"],
        "provider_type": ptype,
        "config": config,
        "name": row.get("name"),
    }
