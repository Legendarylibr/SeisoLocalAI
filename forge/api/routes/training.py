"""Training job routes."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from forge.api.deps import get_db, get_training_orchestrator
from forge.api.routes._stream import spawn_background
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.training import TrainingOrchestrator
from forge.security.audit import audit_event
from forge.security.auth import get_current_user_id
from forge.services.hf_hub import search_huggingface_datasets
from forge.services.jobs import assert_job_owner
from forge.services.models import list_trainable_models, resolve_training_model_id
from forge.services.user_paths import assert_user_training_config
from seiso.models.hf_env import configure_hf_hub_cache
from seiso.security import SecurityError

router = APIRouter(prefix="/training", tags=["training"])


class TrainingStartRequest(BaseModel):
    config: dict
    project_id: str | None = None
    multi_gpu: bool = False
    export_on_complete: dict | None = Field(
        default=None,
        description="Auto-export after training: formats, profile, gguf_quantizations, hub",
    )


class TrainingJobResponse(BaseModel):
    job_id: str
    status: str


def _serialize_metrics_payload(
    points: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    training = [p for p in points if p.get("type") in ("training", "eval")]
    system = [p for p in points if p.get("type") == "system"]
    return {
        "summary": summary or {},
        "training": training[-2000:],
        "system": system[-500:],
        "updated_at": summary.get("updated_at") if summary else None,
    }


@router.get("/datasets")
async def search_datasets(
    user_id: Annotated[str, Depends(get_current_user_id)],
    q: str = Query("", description="Search Hugging Face datasets"),
    limit: int = Query(12, ge=1, le=25),
) -> dict:
    datasets = search_huggingface_datasets(query=q, limit=limit)
    return {"datasets": datasets, "total": len(datasets)}


@router.get("/models")
async def list_training_models(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    inventory = await db.list_models(user_id)
    models = list_trainable_models(inventory, data_dir=settings.data_dir, user_id=user_id)
    return {"models": models, "total": len(models)}


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
    try:
        assert_user_training_config(settings.data_dir, user_id, body.config)
    except SecurityError as exc:
        raise HTTPException(403, str(exc)) from exc

    configure_hf_hub_cache(settings.data_dir)
    inventory = await db.list_models(user_id)
    resolved_model_id, local_path = resolve_training_model_id(
        str(body.config.get("model_id", "")),
        data_dir=settings.data_dir,
        user_id=user_id,
        inventory=inventory,
    )
    original_model_id = str(body.config.get("model_id", ""))
    training_config = {**body.config, "model_id": resolved_model_id}
    training_config.setdefault("extra", {})
    if local_path or resolved_model_id != original_model_id:
        training_config["extra"]["resolved_model_path"] = local_path or resolved_model_id
        training_config["extra"]["original_model_id"] = original_model_id

    job_id = str(uuid.uuid4())
    await db.create_training_job(user_id, training_config, body.project_id, job_id=job_id)
    orchestrator.create_job(job_id=job_id, user_id=user_id)
    payload = {
        "config": {**training_config, "extra": {**training_config.get("extra", {}), "user_id": user_id}},
        "output_dir": str(settings.checkpoints_dir / user_id / job_id),
        "multi_gpu": body.multi_gpu,
        "user_id": user_id,
    }

    async def persist_metrics(
        jid: str,
        points: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> None:
        await db.update_training_metrics(jid, _serialize_metrics_payload(points, summary))

    orchestrator.set_metrics_persister(job_id, persist_metrics)

    async def _run() -> None:
        try:
            await orchestrator.start(job_id, payload)
            job = await orchestrator.wait_for(job_id)
            if job:
                metrics_payload = _serialize_metrics_payload(
                    orchestrator.get_metrics(job_id),
                    job.result.get("metrics_summary"),
                )
                await db.update_job_status(
                    job_id,
                    job.status.value,
                    checkpoint_path=job.result.get("checkpoint_path"),
                    metrics=metrics_payload,
                )
                if job.status.value == "completed" and job.result.get("checkpoint_path"):
                    from forge.services.model_registry import (
                        register_export_outputs,
                        register_training_checkpoint,
                    )

                    await register_training_checkpoint(
                        db,
                        user_id=user_id,
                        data_dir=settings.data_dir,
                        checkpoint_path=job.result["checkpoint_path"],
                        job_id=job_id,
                    )

                    if body.export_on_complete:
                        from forge.api.routes.export import (
                            HubPublishRequest,
                            _hub_metadata_from_request,
                            _resolve_token,
                        )
                        from seiso.export.pipeline import auto_export_after_training

                        export_cfg = dict(body.export_on_complete)
                        hub_req = export_cfg.pop("hub", None)
                        hub_repo = None
                        hub_metadata = None
                        hub_token = None
                        if hub_req:
                            hub = HubPublishRequest(**hub_req)
                            hub_metadata = _hub_metadata_from_request(hub, job_id=job_id, source="training")
                            hub_metadata.validate()
                            hub_repo = hub_metadata.repo_id
                            hub_token = _resolve_token(settings, user_id, hub)

                        export_dir = settings.data_dir / "exports" / user_id / f"train-{job_id}"
                        export_cfg.update(
                            {
                                "hub_repo": hub_repo,
                                "hub_token": hub_token,
                                "hub_metadata": hub_metadata.__dict__ if hub_metadata else None,
                            }
                        )
                        outputs = auto_export_after_training(
                            Path(job.result["checkpoint_path"]),
                            export_dir,
                            export_cfg,
                            sandbox_root=settings.data_dir,
                        )
                        await register_export_outputs(
                            db,
                            user_id=user_id,
                            data_dir=settings.data_dir,
                            outputs={k: str(v) for k, v in outputs.items()},
                            job_id=f"train-{job_id}",
                        )
        except Exception:
            await db.update_job_status(job_id, "failed")

    spawn_background(_run())
    audit_event("training_start", user_id=user_id, job_id=job_id, model_id=body.config.get("model_id"))
    return TrainingJobResponse(job_id=job_id, status="pending")


@router.get("/jobs/{job_id}/metrics")
async def get_training_metrics(
    job_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[TrainingOrchestrator, Depends(get_training_orchestrator)],
) -> dict:
    if not await db.get_training_job(job_id, user_id):
        raise HTTPException(404, "Job not found")
    assert_job_owner(orchestrator, job_id, user_id)

    live = orchestrator.get_metrics(job_id)
    if live:
        return _serialize_metrics_payload(live)

    row = await db.get_training_job(job_id, user_id)
    raw = row.get("metrics_json") if row else None
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return {"summary": {}, "training": [], "system": []}


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
        queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()

        async def forward_logs() -> None:
            try:
                async for line in orchestrator.stream_logs(job_id):
                    await queue.put(("log", line))
            finally:
                await queue.put(None)

        async def forward_metrics() -> None:
            try:
                async for metric in orchestrator.stream_metrics(job_id):
                    await queue.put(("metric", json.dumps(metric)))
            finally:
                await queue.put(None)

        log_task = asyncio.create_task(forward_logs())
        metric_task = asyncio.create_task(forward_metrics())
        finished = 0
        while finished < 2:
            item = await queue.get()
            if item is None:
                finished += 1
                continue
            event, data = item
            yield {"event": event, "data": data}

        await asyncio.gather(log_task, metric_task, return_exceptions=True)

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
