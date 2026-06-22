"""Resolve model inventory IDs to filesystem paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from forge.api.http_errors import raise_forbidden
from forge.db.store import Database
from forge.services.user_paths import assert_user_path, is_local_filesystem_path
from seiso.models.trainable_snapshot import snapshot_has_trainable_weights
from seiso.security import SecurityError

_TRAINABLE_FORMATS = frozenset({"safetensors", "bin", ""})


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
            raise_forbidden(exc)
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
        raise_forbidden(exc)
    if not p.exists():
        raise HTTPException(404, f"Model file missing on disk: {match['path']}")
    return str(p)


def _model_metadata(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata_json")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def resolve_training_model_id(
    model_id: str,
    *,
    data_dir: Path,
    user_id: str,
    inventory: list[dict[str, Any]] | None = None,
) -> tuple[str, str | None]:
    """Resolve a training model ref to a local snapshot path when cached, else HF repo id."""
    if is_local_filesystem_path(model_id):
        try:
            path = assert_user_path(data_dir, user_id, model_id)
            if path.exists() and snapshot_has_trainable_weights(path):
                resolved = str(path.resolve())
                return resolved, resolved
        except SecurityError:
            pass

    hf_source = f"hf:{model_id}"
    for row in inventory or []:
        fmt = (row.get("format") or "").lower()
        if fmt == "gguf":
            continue
        source = row.get("source") or ""
        meta = _model_metadata(row)
        repo_match = source == hf_source or meta.get("repo_id") == model_id
        if not repo_match:
            continue
        try:
            path = assert_user_path(data_dir, user_id, row["path"])
        except SecurityError:
            continue
        if path.exists() and snapshot_has_trainable_weights(path):
            resolved = str(path.resolve())
            return resolved, resolved

    return model_id, None


def list_trainable_models(
    inventory: list[dict[str, Any]],
    *,
    data_dir: Path,
    user_id: str,
) -> list[dict[str, Any]]:
    """Inventory rows suitable as LoRA/SFT base models."""
    options: list[dict[str, Any]] = []
    for row in inventory:
        fmt = (row.get("format") or "").lower()
        if fmt and fmt not in _TRAINABLE_FORMATS:
            continue
        if fmt == "gguf":
            continue
        try:
            path = assert_user_path(data_dir, user_id, row["path"])
        except SecurityError:
            continue
        if not path.exists():
            continue
        if not snapshot_has_trainable_weights(path):
            continue
        meta = _model_metadata(row)
        repo_id = meta.get("repo_id")
        if not repo_id and isinstance(row.get("source"), str) and row["source"].startswith("hf:"):
            repo_id = row["source"][3:]
        options.append(
            {
                "id": row["id"],
                "name": row["name"],
                "path": str(path.resolve()),
                "repo_id": repo_id,
                "source": row.get("source"),
                "format": fmt or "safetensors",
                "size_bytes": row.get("size_bytes", 0),
            }
        )
    return options
