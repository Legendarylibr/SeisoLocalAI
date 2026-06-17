"""Export job routes."""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from forge.api.deps import get_db, get_export_orchestrator
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.export import ExportOrchestrator
from forge.security.audit import audit_event
from forge.security.auth import get_current_user_id
from forge.services.user_paths import assert_user_path
from seiso.security import SecurityError

router = APIRouter(prefix="/export", tags=["export"])


class ExportStartRequest(BaseModel):
    checkpoint: str
    formats: list[str] = Field(default_factory=lambda: ["merged"])
    gguf_quantizations: list[str] = Field(default_factory=lambda: ["q4_k_m"])
    hub_repo: str | None = None


class ExportJobResponse(BaseModel):
    job_id: str
    status: str


@router.get("/jobs")
async def list_export_jobs(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> list[dict]:
    return await db.list_export_jobs(user_id)


@router.post("/jobs", response_model=ExportJobResponse)
async def start_export(
    body: ExportStartRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[ExportOrchestrator, Depends(get_export_orchestrator)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> ExportJobResponse:
    try:
        assert_user_path(settings.data_dir, user_id, body.checkpoint)
    except SecurityError as exc:
        raise HTTPException(403, str(exc)) from exc

    job_id = str(uuid.uuid4())
    await db.create_export_job(user_id, body.model_dump(), job_id=job_id)
    orchestrator.create_job(job_id=job_id, user_id=user_id)
    payload = {
        **body.model_dump(),
        "user_id": user_id,
        "output_dir": str(settings.data_dir / "exports" / user_id / job_id),
        "hub_token": settings.hf_token or None,
    }

    async def _run() -> None:
        try:
            await orchestrator.start(job_id, payload)
            job = await orchestrator.wait_for(job_id)
            if job:
                await db.update_export_job_status(
                    job_id,
                    job.status.value,
                    output_paths=job.result.get("outputs"),
                )
        except Exception:
            await db.update_export_job_status(job_id, "failed")
            raise

    asyncio.create_task(_run())
    audit_event("export_start", user_id=user_id, job_id=job_id, formats=body.formats)
    return ExportJobResponse(job_id=job_id, status="pending")


@router.get("/jobs/{job_id}/stream")
async def stream_export(
    job_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[ExportOrchestrator, Depends(get_export_orchestrator)],
):
    if not await db.get_export_job(job_id, user_id):
        raise HTTPException(404, "Job not found")
    assert_job_owner(orchestrator, job_id, user_id)

    async def event_gen():
        async for line in orchestrator.stream_logs(job_id):
            yield {"event": "log", "data": line}
        j = orchestrator.get_job(job_id)
        if j and j.error:
            yield {"event": "error", "data": j.error}
        if j and j.result:
            yield {"event": "result", "data": str(j.result)}

    return EventSourceResponse(event_gen())
