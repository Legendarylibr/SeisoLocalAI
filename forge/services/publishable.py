"""Validate that only Seiso-created outputs may be published to Hugging Face."""

from __future__ import annotations

import json
from pathlib import Path

from forge.db.store import Database
from forge.services.user_paths import assert_user_path

# Sources that represent models created inside Seiso (not hub downloads or manual scans).
PUSHABLE_SOURCES = frozenset({"training", "export"})


async def get_model_for_user(db: Database, model_id: str, user_id: str) -> dict | None:
    return await db.get_model(model_id, user_id)


def _parse_metadata(model: dict) -> dict:
    raw = model.get("metadata_json") or "{}"
    try:
        return json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    except json.JSONDecodeError:
        return {}


def is_pushable_model(model: dict) -> bool:
    source = (model.get("source") or "").split(":")[0]
    return source in PUSHABLE_SOURCES


async def assert_pushable_model(db: Database, *, model_id: str, user_id: str) -> dict:
    model = await get_model_for_user(db, model_id, user_id)
    if not model:
        raise ValueError("Model not found")
    if not is_pushable_model(model):
        raise ValueError("Only Seiso training or export outputs can be published to Hugging Face")
    return model


async def assert_pushable_path(
    db: Database,
    *,
    data_dir: Path,
    user_id: str,
    target: str | Path,
) -> Path:
    """Path must be a registered pushable model or under exports/{user_id}/ from a completed job."""
    resolved = assert_user_path(data_dir, user_id, target)
    rel = resolved.relative_to(data_dir.resolve())
    if rel.parts[:2] == ("exports", user_id):
        return resolved

    exact = await db.get_model_by_path(user_id, str(resolved))
    if exact and is_pushable_model(exact):
        return resolved
    parent = await db.get_model_by_path(user_id, str(resolved.parent))
    if parent and is_pushable_model(parent):
        return resolved
    # Walk parents for nested export artifacts under a registered model path.
    cursor = resolved.parent
    data_root = data_dir.resolve()
    while True:
        try:
            cursor.relative_to(data_root)
        except ValueError:
            break
        if cursor == data_root:
            break
        row = await db.get_model_by_path(user_id, str(cursor))
        if row and is_pushable_model(row):
            return resolved
        if cursor.parent == cursor:
            break
        cursor = cursor.parent

    raise ValueError("Only Seiso export outputs can be published to Hugging Face")


async def assert_pushable_checkpoint(
    db: Database,
    *,
    data_dir: Path,
    user_id: str,
    checkpoint: str | Path,
) -> Path:
    """Checkpoint must be a training checkpoint or prior pushable export for fine-tune export."""
    resolved = assert_user_path(data_dir, user_id, checkpoint)
    rel = resolved.relative_to(data_dir.resolve())

    if rel.parts[:2] == ("checkpoints", user_id):
        return resolved

    norm = str(resolved.resolve())
    exact = await db.get_model_by_path(user_id, str(resolved))
    if exact and is_pushable_model(exact):
        return resolved

    models = await db.list_models(user_id)
    for m in models:
        if str(Path(m["path"]).resolve()) == norm and is_pushable_model(m):
            return resolved

    raise ValueError("Checkpoint must be a Seiso training checkpoint or a prior export output")


async def list_publishable_models(db: Database, user_id: str) -> list[dict]:
    models = await db.list_models(user_id)
    out = []
    for m in models:
        if not is_pushable_model(m):
            continue
        meta = _parse_metadata(m)
        out.append(
            {
                "id": m["id"],
                "name": m["name"],
                "path": m["path"],
                "source": m.get("source"),
                "format": m.get("format"),
                "size_bytes": m.get("size_bytes", 0),
                "job_id": meta.get("job_id"),
                "export_key": meta.get("export_key"),
            }
        )
    return out
