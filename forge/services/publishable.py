"""Validate that only Seiso-created outputs may be published to Hugging Face."""

from __future__ import annotations

import json
from pathlib import Path

from forge.db.store import Database
from forge.services.user_paths import assert_user_path

# Sources that represent models created inside Seiso (not hub downloads or manual scans).
PUSHABLE_SOURCES = frozenset({"training", "export", "rl_quant"})


async def get_model_for_user(db: Database, model_id: str, user_id: str) -> dict | None:
    models = await db.list_models(user_id)
    return next((m for m in models if m["id"] == model_id), None)


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
        raise ValueError(
            "Only Seiso training, export, or RL quant outputs can be published to Hugging Face"
        )
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

    models = await db.list_models(user_id)
    for model in models:
        if not is_pushable_model(model):
            continue
        model_path = Path(model["path"]).resolve()
        try:
            resolved.relative_to(model_path)
            return resolved
        except ValueError:
            pass
        if model_path == resolved or model_path == resolved.parent:
            return resolved

    rel = resolved.relative_to(data_dir.resolve())
    if rel.parts[:2] == ("exports", user_id):
        return resolved

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

    models = await db.list_models(user_id)
    norm = str(resolved.resolve())
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
