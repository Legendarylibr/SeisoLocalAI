"""Resolve model inventory IDs to filesystem paths."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from forge.db.store import Database
from forge.services.user_paths import assert_user_path
from seiso.security import SecurityError


async def resolve_model_path(
    db: Database,
    user_id: str,
    *,
    model_id: str | None,
    model_path: str | None,
    data_dir: Path,
) -> str | None:
    """Return a validated local model path scoped to the user, or None."""
    if model_path:
        try:
            p = assert_user_path(data_dir, user_id, model_path)
        except SecurityError as exc:
            raise HTTPException(403, str(exc)) from exc
        if not p.exists():
            raise HTTPException(404, f"Model path not found: {model_path}")
        return str(p)

    if not model_id:
        return None

    models = await db.list_models(user_id)
    match = next((m for m in models if m["id"] == model_id or m["name"] == model_id), None)
    if not match:
        raise HTTPException(404, f"Model not found in inventory: {model_id}")

    try:
        p = assert_user_path(data_dir, user_id, match["path"])
    except SecurityError as exc:
        raise HTTPException(403, str(exc)) from exc
    if not p.exists():
        raise HTTPException(404, f"Model file missing on disk: {match['path']}")
    return str(p)
