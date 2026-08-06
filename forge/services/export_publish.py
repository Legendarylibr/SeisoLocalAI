"""Hub publish folder resolution for export routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from forge.api.schemas.export import PublishToHubRequest
from forge.db.store import Database
from forge.services.publishable import (
    assert_pushable_model,
    assert_pushable_path,
)
from seiso.security import SecurityError


def loads_json_field(raw: Any, fallback: Any) -> Any:
    if not raw:
        return fallback
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


async def resolve_publish_folder(
    body: PublishToHubRequest,
    *,
    user_id: str,
    db: Database,
    data_dir: Path,
) -> tuple[Path, str | None, str]:
    """Return (folder, job_id, source) for a publish request."""
    if body.model_id:
        model = await assert_pushable_model(db, model_id=body.model_id, user_id=user_id)
        try:
            folder = await assert_pushable_path(
                db,
                data_dir=data_dir,
                user_id=user_id,
                target=model["path"],
            )
        except (SecurityError, ValueError) as exc:
            raise HTTPException(403 if isinstance(exc, SecurityError) else 400, str(exc)) from exc
        meta_raw = loads_json_field(model.get("metadata_json") or "{}", {})
        job_id = meta_raw.get("job_id")
        source = model.get("source") or "export"
    elif body.export_job_id:
        job = await db.get_export_job(body.export_job_id, user_id)
        if not job or job.get("status") != "completed":
            raise HTTPException(400, "Export job not found or not completed")
        outputs = loads_json_field(job.get("output_paths_json") or "{}", {})
        if not outputs:
            raise HTTPException(400, "Export job has no outputs")
        preferred = next((v for k, v in outputs.items() if "gguf" in k.lower()), None)
        if not preferred:
            preferred = outputs.get("merged") or next(iter(outputs.values()))
        try:
            folder = await assert_pushable_path(
                db,
                data_dir=data_dir,
                user_id=user_id,
                target=preferred,
            )
        except (SecurityError, ValueError) as exc:
            raise HTTPException(403 if isinstance(exc, SecurityError) else 400, str(exc)) from exc
        job_id = body.export_job_id
        source = "export"
    elif body.output_path:
        try:
            folder = await assert_pushable_path(
                db, data_dir=data_dir, user_id=user_id, target=body.output_path
            )
        except (SecurityError, ValueError) as exc:
            raise HTTPException(403 if isinstance(exc, SecurityError) else 400, str(exc)) from exc
        job_id = None
        source = "export"
    else:
        raise HTTPException(400, "Provide model_id, export_job_id, or output_path")

    if folder.is_file():
        folder = folder.parent
    return folder, job_id, source
