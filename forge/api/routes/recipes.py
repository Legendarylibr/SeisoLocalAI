"""Recipe workflow routes."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from forge.api.deps import get_recipe_orchestrator
from forge.orchestrators.recipes import RecipeOrchestrator
from forge.security.auth import get_current_user_id
from forge.services.jobs import assert_job_owner

router = APIRouter(prefix="/recipes", tags=["recipes"])


class RecipeRunRequest(BaseModel):
    recipe: dict


@router.post("/jobs")
async def run_recipe(
    body: RecipeRunRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    orchestrator: Annotated[RecipeOrchestrator, Depends(get_recipe_orchestrator)],
) -> dict:
    job_id = orchestrator.create_job(user_id=user_id)
    asyncio.create_task(orchestrator.start(job_id, {**body.model_dump(), "user_id": user_id}))
    return {"job_id": job_id, "status": "pending"}


@router.get("/jobs/{job_id}/stream")
async def stream_recipe(
    job_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    orchestrator: Annotated[RecipeOrchestrator, Depends(get_recipe_orchestrator)],
):
    assert_job_owner(orchestrator, job_id, user_id)

    async def event_gen():
        async for line in orchestrator.stream_logs(job_id):
            yield {"event": "log", "data": line}
        j = orchestrator.get_job(job_id)
        if j and j.error:
            yield {"event": "error", "data": j.error}

    return EventSourceResponse(event_gen())
