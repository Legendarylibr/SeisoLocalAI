"""Training job routes."""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from forge.api.deps import get_db, get_training_orchestrator
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.training import TrainingOrchestrator
from forge.security.audit import audit_event
from forge.security.auth import get_current_user_id
from forge.services.jobs import assert_job_owner

router = APIRouter(prefix="/training", tags=["training"])


class TrainingStartRequest(BaseModel):
    config: dict
    project_id: str | None = None
    multi_gpu: bool = False


class TrainingJobResponse(BaseModel):
    job_id: str
    status: str


@router.get("/jobs")
async def list_jobs(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> list[dict]:
    return await db.list_training_jobs(user_id)


@router.post("/jobs", response_model=TrainingJobResponse)
async def start_training(
    body: TrainingStartRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[TrainingOrchestrator, Depends(get_training_orchestrator)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> TrainingJobResponse:
    job_id = str(uuid.uuid4())
    await db.create_training_job(user_id, body.config, body.project_id, job_id=job_id)
    orchestrator.create_job(job_id=job_id, user_id=user_id)
    payload = {
        "config": body.config,
        "output_dir": str(settings.checkpoints_dir / user_id / job_id),
        "multi_gpu": body.multi_gpu,
        "user_id": user_id,
    }

    async def _run() -> None:
        try:
            await orchestrator.start(job_id, payload)
            job = await orchestrator.wait_for(job_id)
            if job:
                await db.update_job_status(
                    job_id,
                    job.status.value,
                    checkpoint_path=job.result.get("checkpoint_path"),
                )
        except Exception as exc:
            await db.update_job_status(job_id, "failed")
            raise exc

    asyncio.create_task(_run())
    audit_event("training_start", user_id=user_id, job_id=job_id, model_id=body.config.get("model_id"))
    return TrainingJobResponse(job_id=job_id, status="pending")


@router.get("/jobs/{job_id}/stream")
async def stream_training(
    job_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[TrainingOrchestrator, Depends(get_training_orchestrator)],
):
    if not await db.get_training_job(job_id, user_id):
        raise HTTPException(404, "Job not found")
    assert_job_owner(orchestrator, job_id, user_id)

    async def event_gen():
        async for line in orchestrator.stream_logs(job_id):
            yield {"event": "log", "data": line}
        j = orchestrator.get_job(job_id)
        if j and j.error:
            yield {"event": "error", "data": j.error}
        yield {"event": "status", "data": j.status.value if j else "unknown"}

    return EventSourceResponse(event_gen())


@router.post("/jobs/{job_id}/cancel")
async def cancel_training(
    job_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[TrainingOrchestrator, Depends(get_training_orchestrator)],
) -> dict:
    if not await db.get_training_job(job_id, user_id):
        raise HTTPException(404, "Job not found")
    assert_job_owner(orchestrator, job_id, user_id)
    ok = await orchestrator.cancel(job_id)
    return {"cancelled": ok}
