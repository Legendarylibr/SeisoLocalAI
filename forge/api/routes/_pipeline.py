"""Factory for stage-based pipeline job routers (compress)."""

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from forge.api.deps import get_db
from forge.api.routes._jobs import format_stage_pipeline_job, stage_presets_response
from forge.api.routes._stream import (
    durable_job_events,
    job_log_event_gen,
    spawn_background,
)
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.base import Orchestrator
from forge.security.audit import audit_event
from forge.security.auth import get_current_user_id
from forge.services.job_runtime import run_orchestrated_job
from forge.services.jobs import assert_job_owner
from forge.services.model_registry import register_export_outputs


class PipelineJobResponse(BaseModel):
    job_id: str
    status: str


@dataclass(frozen=True)
class FormattedJobRoutes:
    format_job: Callable[[dict], dict]
    list_jobs: Callable[[Database, str], Awaitable[list[dict]]]
    get_job: Callable[[Database, str, str], Awaitable[dict | None]]
    get_orchestrator: Callable[[], Orchestrator]
    before_result: Callable[[dict[str, Any]], list[dict[str, str]]] | None = None


def register_formatted_job_routes(
    router: APIRouter, routes: FormattedJobRoutes
) -> None:
    """Register GET /jobs, GET /jobs/{id}, and GET /jobs/{id}/stream on a router."""
    get_orch = routes.get_orchestrator

    @router.get("/jobs")
    async def list_jobs(
        user_id: Annotated[str, Depends(get_current_user_id)],
        db: Annotated[Database, Depends(get_db)],
    ) -> list[dict]:
        rows = await routes.list_jobs(db, user_id)
        return [routes.format_job(row) for row in rows]

    @router.get("/jobs/{job_id}")
    async def get_job_route(
        job_id: str,
        user_id: Annotated[str, Depends(get_current_user_id)],
        db: Annotated[Database, Depends(get_db)],
    ) -> dict:
        job = await routes.get_job(db, job_id, user_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return routes.format_job(job)

    @router.get("/jobs/{job_id}/stream")
    async def stream_job(
        job_id: str,
        user_id: Annotated[str, Depends(get_current_user_id)],
        db: Annotated[Database, Depends(get_db)],
        orchestrator: Annotated[Orchestrator, Depends(get_orch)],
    ):
        if not await routes.get_job(db, job_id, user_id):
            raise HTTPException(404, "Job not found")
        if orchestrator.get_job(job_id):
            assert_job_owner(orchestrator, job_id, user_id)
            return EventSourceResponse(
                job_log_event_gen(
                    orchestrator,
                    job_id,
                    db=db,
                    user_id=user_id,
                    before_result=routes.before_result,
                )
            )

        row = await routes.get_job(db, job_id, user_id)

        async def db_event_gen():
            async for event in durable_job_events(db, job_id, user_id):
                yield event
            if row and row.get("error_text"):
                yield {"event": "error", "data": row["error_text"]}
            # Default schema stores stage_results_json='{}' — only emit a real
            # result for successful terminal jobs with non-empty payloads.
            stage_json = (row or {}).get("stage_results_json") or ""
            status = str((row or {}).get("status") or "").lower()
            if (
                status == "completed"
                and stage_json.strip()
                and stage_json.strip() not in {"{}", "null", "None"}
            ):
                yield {"event": "result", "data": stage_json}

        return EventSourceResponse(db_event_gen())


@dataclass(frozen=True)
class StagePipelineRouterConfig:
    prefix: str
    tags: tuple[str, ...]
    audit_event_name: str
    export_registry_key: str
    presets: dict[str, dict[str, Any]]
    stage_order: tuple[str, ...] | list[str]
    stage_help: dict[str, str]
    start_request_model: type[BaseModel]
    get_orchestrator: Callable[[], Orchestrator]
    list_jobs: Callable[[Database, str], Awaitable[list[dict]]]
    get_job: Callable[[Database, str, str], Awaitable[dict | None]]
    create_job: Callable[[Database, str, dict, str], Awaitable[Any]]
    update_status: Callable[..., Awaitable[None]]  # (db, job_id, status, **fields)
    prepare_config: Callable[[BaseModel, Database, str, ForgeSettings], Awaitable[dict]]
    enrich_stage_results: Callable[[dict[str, Any]], dict[str, Any]]
    model_defaults: dict[str, Any] | None = None


def build_stage_pipeline_router(config: StagePipelineRouterConfig) -> APIRouter:
    router = APIRouter(prefix=config.prefix, tags=list(config.tags))
    StartRequest = config.start_request_model

    register_formatted_job_routes(
        router,
        FormattedJobRoutes(
            format_job=format_stage_pipeline_job,
            list_jobs=config.list_jobs,
            get_job=config.get_job,
            get_orchestrator=config.get_orchestrator,
        ),
    )

    get_orch = config.get_orchestrator

    @router.post("/jobs", response_model=PipelineJobResponse)
    async def start_job(
        body: Annotated[StartRequest, Body()],
        user_id: Annotated[str, Depends(get_current_user_id)],
        db: Annotated[Database, Depends(get_db)],
        orchestrator: Annotated[Orchestrator, Depends(get_orch)],
        settings: Annotated[ForgeSettings, Depends(get_settings)],
    ) -> PipelineJobResponse:
        job_id = str(uuid.uuid4())
        config_payload = await config.prepare_config(body, db, user_id, settings)

        await config.create_job(db, user_id, config_payload, job_id)
        orchestrator.create_job(job_id=job_id, user_id=user_id)
        payload = {**config_payload, "user_id": user_id}

        async def _finished(job) -> None:
            result = job.result or {}
            stage_results = config.enrich_stage_results(result)
            await config.update_status(
                db,
                job_id,
                job.status.value,
                output_dir=result.get("output_root"),
                run_dir=result.get("run_dir"),
                model_dir=result.get("model_dir"),
                stages=result.get("stages"),
                stage_results=stage_results,
                error_text=job.error if job.status.value == "failed" else None,
            )
            if model_dir := result.get("model_dir"):
                try:
                    await register_export_outputs(
                        db,
                        user_id=user_id,
                        data_dir=settings.data_dir,
                        outputs={config.export_registry_key: str(model_dir)},
                        job_id=job_id,
                    )
                except Exception:
                    import logging

                    logging.getLogger(__name__).exception(
                        "Pipeline inventory registration failed for job %s "
                        "(job remains %s)",
                        job_id,
                        job.status.value,
                    )

        async def _failed(message: str) -> None:
            await config.update_status(db, job_id, "failed", error_text=message)

        async def _run() -> None:
            await run_orchestrated_job(
                orchestrator=orchestrator,
                job_id=job_id,
                payload=payload,
                on_finished=_finished,
                on_failed=_failed,
            )

        spawn_background(_run())
        preset = config_payload.get("preset", "")
        audit_event(
            config.audit_event_name, user_id=user_id, job_id=job_id, preset=preset
        )
        return PipelineJobResponse(job_id=job_id, status="pending")

    @router.get("/presets")
    async def list_presets(
        user_id: Annotated[str, Depends(get_current_user_id)],
    ) -> dict[str, Any]:
        return stage_presets_response(
            config.presets,
            config.stage_order,
            config.stage_help,
            config.model_defaults,
        )

    return router
