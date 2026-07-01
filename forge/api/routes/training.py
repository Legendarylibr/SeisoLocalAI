"""Training job routes."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from forge.api.deps import get_db, get_training_orchestrator
from forge.api.http_errors import raise_forbidden
from forge.api.routes._stream import job_failure_message, spawn_background
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.training import TrainingOrchestrator
from forge.security.audit import audit_event
from forge.security.auth import get_current_user_id
from forge.services.hardware import hardware_profile
from forge.services.hf_hub import search_huggingface_datasets
from forge.services.jobs import assert_job_owner
from forge.services.models import list_trainable_models, resolve_training_model_id
from forge.services.user_paths import (
    assert_user_training_config,
    resolve_training_dataset_path,
)
from seiso.models.hf_env import configure_hf_hub_cache
from seiso.models.hub_quant import native_quant_training_block_reason
from seiso.models.trainable_snapshot import is_gguf_only_repo_id
from seiso.security import SecurityError
from seiso.training.config import DatasetFormat, TrainConfig
from seiso.training.dataset_analysis import analyze_training_dataset
from seiso.training.recommendations import recommend_training_config

router = APIRouter(prefix="/training", tags=["training"])


class TrainingStartRequest(BaseModel):
    config: dict
    project_id: str | None = None
    multi_gpu: bool = False
    dataset_analysis_token: str | None = None
    export_on_complete: dict | None = Field(
        default=None,
        description="Auto-export after training: formats, profile, gguf_quantizations, hub",
    )


class TrainingJobResponse(BaseModel):
    job_id: str
    status: str


_DATASET_ANALYSIS_TTL_S = 10 * 60.0


@dataclass(frozen=True)
class _DatasetAnalysisCacheEntry:
    user_id: str
    dataset: str
    requested_format: str
    resolved_format: str
    valid: bool
    created_at: float


_dataset_analysis_cache: dict[str, _DatasetAnalysisCacheEntry] = {}


def _store_dataset_analysis_token(
    *,
    user_id: str,
    dataset: str | Path,
    requested_format: DatasetFormat,
    resolved_format: str | None,
    valid: bool,
) -> str:
    token = uuid.uuid4().hex
    _dataset_analysis_cache[token] = _DatasetAnalysisCacheEntry(
        user_id=user_id,
        dataset=str(dataset),
        requested_format=requested_format.value,
        resolved_format=resolved_format or requested_format.value,
        valid=valid,
        created_at=time.monotonic(),
    )
    return token


def _dataset_analysis_token_matches(
    token: str | None,
    *,
    user_id: str,
    dataset: str | Path,
    dataset_format: DatasetFormat,
) -> bool:
    if not token:
        return False
    entry = _dataset_analysis_cache.get(token)
    now = time.monotonic()
    if entry is None:
        return False
    if now - entry.created_at > _DATASET_ANALYSIS_TTL_S:
        _dataset_analysis_cache.pop(token, None)
        return False
    return (
        entry.user_id == user_id
        and entry.dataset == str(dataset)
        and dataset_format.value in {entry.requested_format, entry.resolved_format}
        and entry.valid
    )


def _serialize_metrics_payload(
    points: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from seiso.security.hardware_privacy import sanitize_system_metric_point

    training = [p for p in points if p.get("type") in ("training", "eval")]
    system = [
        sanitize_system_metric_point(p) for p in points if p.get("type") == "system"
    ]
    return {
        "summary": summary or {},
        "training": training[-2000:],
        "system": system[-500:],
        "updated_at": summary.get("updated_at") if summary else None,
    }


def _effective_job_status(
    db_status: str,
    orchestrator: TrainingOrchestrator,
    job_id: str,
) -> str:
    live = orchestrator.get_job(job_id)
    if not live:
        return db_status
    live_status = live.status.value
    if db_status == "pending" and live_status in (
        "running",
        "completed",
        "failed",
        "cancelled",
    ):
        return live_status
    return db_status


def _format_training_job_row(
    row: dict[str, Any],
    orchestrator: TrainingOrchestrator | None = None,
) -> dict[str, Any]:
    job_id = str(row.get("id", ""))
    status = str(row.get("status", "unknown"))
    if orchestrator is not None:
        status = _effective_job_status(status, orchestrator, job_id)
    return {
        "id": job_id,
        "status": status,
        "config_json": row.get("config_json"),
        "metrics_json": row.get("metrics_json"),
        "error_text": row.get("error_text"),
        "checkpoint_path": row.get("checkpoint_path"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "project_id": row.get("project_id"),
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
    models = list_trainable_models(
        inventory, data_dir=settings.data_dir, user_id=user_id
    )
    return {"models": models, "total": len(models)}


@router.get("/recommendations")
async def training_recommendations(
    user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
    model_id: str = Query("", description="Base model repo id or local path"),
    dataset: str = Query("", description="Dataset hub id or local path"),
) -> dict:
    profile = hardware_profile()
    resolved_dataset = dataset
    if isinstance(dataset, str) and dataset.strip():
        with contextlib.suppress(Exception):
            resolved_dataset = resolve_training_dataset_path(
                settings.data_dir,
                user_id,
                dataset,
                install_root=Path(__file__).resolve().parents[3],
            )
    return recommend_training_config(
        profile,
        model_id=model_id,
        dataset=str(resolved_dataset),
        sandbox_root=Path(settings.data_dir) / "uploads" / user_id,
    )


class DatasetValidationRequest(BaseModel):
    dataset: str
    dataset_format: str = "auto"


def _resolve_dataset_for_user(
    dataset: str,
    *,
    user_id: str,
    settings: ForgeSettings,
) -> str | Path:
    if not dataset or not isinstance(dataset, str):
        return dataset
    return resolve_training_dataset_path(
        settings.data_dir,
        user_id,
        dataset,
        install_root=Path(__file__).resolve().parents[3],
    )


def _run_dataset_analysis(
    dataset: str | Path,
    *,
    dataset_format: DatasetFormat,
    sandbox_root: Path,
) -> dict[str, Any]:
    return analyze_training_dataset(
        dataset,
        dataset_format=dataset_format,
        sandbox_root=sandbox_root,
    )


@router.post("/analyze-dataset")
async def analyze_dataset_endpoint(
    body: DatasetValidationRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    """Research-grade dataset analysis: full-corpus schema detection and training hints."""
    ds = _resolve_dataset_for_user(body.dataset, user_id=user_id, settings=settings)
    try:
        ds_fmt = (
            DatasetFormat(body.dataset_format)
            if body.dataset_format
            else DatasetFormat.AUTO
        )
        analysis = await asyncio.to_thread(
            _run_dataset_analysis,
            ds,
            dataset_format=ds_fmt,
            sandbox_root=Path(settings.data_dir) / "uploads" / user_id,
        )
        analysis["analysis_token"] = _store_dataset_analysis_token(
            user_id=user_id,
            dataset=ds,
            requested_format=ds_fmt,
            resolved_format=analysis.get("resolved_format"),
            valid=bool(analysis.get("valid")),
        )
        return analysis
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/validate-dataset")
async def validate_dataset_endpoint(
    body: DatasetValidationRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    """Preflight endpoint — scans the entire dataset before training starts."""
    ds = _resolve_dataset_for_user(body.dataset, user_id=user_id, settings=settings)
    try:
        ds_fmt = (
            DatasetFormat(body.dataset_format)
            if body.dataset_format
            else DatasetFormat.AUTO
        )
        analysis = await asyncio.to_thread(
            _run_dataset_analysis,
            ds,
            dataset_format=ds_fmt,
            sandbox_root=Path(settings.data_dir) / "uploads" / user_id,
        )
        if not analysis.get("valid"):
            raise ValueError("No valid training samples after preprocessing")
        analysis["analysis_token"] = _store_dataset_analysis_token(
            user_id=user_id,
            dataset=ds,
            requested_format=ds_fmt,
            resolved_format=analysis.get("resolved_format"),
            valid=True,
        )
        return {"valid": True, **analysis}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/jobs")
async def list_jobs(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[TrainingOrchestrator, Depends(get_training_orchestrator)],
) -> list[dict]:
    rows = await db.list_training_jobs(user_id)
    return [_format_training_job_row(row, orchestrator) for row in rows]


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[TrainingOrchestrator, Depends(get_training_orchestrator)],
) -> dict:
    row = await db.get_training_job(job_id, user_id)
    if not row:
        raise HTTPException(404, "Job not found")
    return _format_training_job_row(row, orchestrator)


@router.post("/jobs", response_model=TrainingJobResponse)
async def start_training(
    body: TrainingStartRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[TrainingOrchestrator, Depends(get_training_orchestrator)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> TrainingJobResponse:
    training_config = dict(body.config)
    dataset = training_config.get("dataset")
    if isinstance(dataset, str):
        training_config["dataset"] = resolve_training_dataset_path(
            settings.data_dir,
            user_id,
            dataset,
            install_root=Path(__file__).resolve().parents[3],
        )

    try:
        assert_user_training_config(settings.data_dir, user_id, training_config)
    except SecurityError as exc:
        raise_forbidden(exc)

    try:
        TrainConfig.model_validate(training_config)
    except Exception as exc:
        raise HTTPException(400, f"Invalid training configuration: {exc}") from exc

    # Early dataset normalization check — fail fast with clear error *before* queuing the job
    # or downloading the base model. This fulfills "show error before training".
    dataset_for_val = training_config.get("dataset")
    ds_fmt_str = training_config.get("dataset_format", "auto")
    try:
        ds_fmt = DatasetFormat(ds_fmt_str) if ds_fmt_str else DatasetFormat.AUTO
        if not _dataset_analysis_token_matches(
            body.dataset_analysis_token,
            user_id=user_id,
            dataset=dataset_for_val,
            dataset_format=ds_fmt,
        ):
            analysis = await asyncio.to_thread(
                _run_dataset_analysis,
                dataset_for_val,
                dataset_format=ds_fmt,
                sandbox_root=Path(settings.data_dir) / "uploads" / user_id,
            )
            if not analysis.get("valid"):
                raise ValueError("No valid training samples after preprocessing")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset cannot be normalized for training: {exc}",
        ) from exc

    configure_hf_hub_cache(settings.data_dir)
    from forge.services.hf_auth import resolve_hf_token_for_download

    hf_token, _ = resolve_hf_token_for_download(
        user_id=user_id,
        data_dir=settings.data_dir,
        encryption_key=settings.hf_token_encryption_key,
        settings_token=settings.hf_token or None,
    )
    inventory = await db.list_models(user_id)
    original_model_id = str(training_config.get("model_id", ""))
    if original_model_id and is_gguf_only_repo_id(original_model_id):
        from seiso.models.trainable_snapshot import GGUF_ONLY_REPO_MESSAGE

        raise HTTPException(400, GGUF_ONLY_REPO_MESSAGE)
    native_quant_block = native_quant_training_block_reason(original_model_id)
    if native_quant_block:
        raise HTTPException(400, native_quant_block)
    try:
        resolved_model_id, local_path = resolve_training_model_id(
            original_model_id,
            data_dir=settings.data_dir,
            user_id=user_id,
            inventory=inventory,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    training_config = {**training_config, "model_id": resolved_model_id}
    training_config.setdefault("extra", {})
    if local_path or resolved_model_id != original_model_id:
        training_config["extra"]["resolved_model_path"] = (
            local_path or resolved_model_id
        )
        training_config["extra"]["original_model_id"] = original_model_id
    resolved_block = native_quant_training_block_reason(resolved_model_id)
    if resolved_block:
        raise HTTPException(400, resolved_block)

    job_id = str(uuid.uuid4())
    await db.create_training_job(
        user_id, training_config, body.project_id, job_id=job_id
    )
    orchestrator.create_job(job_id=job_id, user_id=user_id)
    payload = {
        "config": {
            **training_config,
            "sandbox_root": str(settings.data_dir / "uploads" / user_id),
            "extra": {**training_config.get("extra", {}), "user_id": user_id},
        },
        "output_dir": str(settings.checkpoints_dir / user_id / job_id),
        "multi_gpu": body.multi_gpu,
        "user_id": user_id,
        "hf_token": hf_token,
    }

    async def persist_metrics(
        jid: str,
        points: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> None:
        await db.update_training_metrics(
            jid, _serialize_metrics_payload(points, summary), user_id=user_id
        )

    orchestrator.set_metrics_persister(job_id, persist_metrics)

    async def _run() -> None:
        try:
            await db.update_job_status(job_id, "running", user_id=user_id)
            await orchestrator.start(job_id, payload)
            job = await orchestrator.wait_for(job_id)
            if job:
                metrics_payload = _serialize_metrics_payload(
                    orchestrator.get_metrics(job_id),
                    job.result.get("metrics_summary"),
                )
                error_text = job.error
                if not error_text and job.status.value == "failed":
                    error_text = job_failure_message(orchestrator, job_id)
                await db.update_job_status(
                    job_id,
                    job.status.value,
                    checkpoint_path=job.result.get("checkpoint_path"),
                    metrics=metrics_payload,
                    error_text=error_text,
                    user_id=user_id,
                )
                if job.status.value == "completed" and job.result.get(
                    "checkpoint_path"
                ):
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
                        from forge.services.hub_publish import (
                            HubPublishRequest,
                            hub_metadata_from_request,
                            resolve_hub_publish_token,
                        )
                        from seiso.export.pipeline import auto_export_after_training

                        export_cfg = dict(body.export_on_complete)
                        hub_req = export_cfg.pop("hub", None)
                        hub_repo = None
                        hub_metadata = None
                        hub_token = None
                        if hub_req:
                            hub = HubPublishRequest(**hub_req)
                            hub_metadata = hub_metadata_from_request(
                                hub, job_id=job_id, source="training"
                            )
                            hub_metadata.validate()
                            hub_repo = hub_metadata.repo_id
                            hub_token = resolve_hub_publish_token(
                                settings, user_id, hub
                            )

                        export_dir = (
                            settings.data_dir / "exports" / user_id / f"train-{job_id}"
                        )
                        export_cfg.update(
                            {
                                "hub_repo": hub_repo,
                                "hub_token": hub_token,
                                "hub_metadata": (
                                    hub_metadata.__dict__ if hub_metadata else None
                                ),
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
        except Exception as exc:
            await db.update_job_status(
                job_id,
                "failed",
                error_text=job_failure_message(orchestrator, job_id, exc),
                user_id=user_id,
            )

    spawn_background(_run())
    audit_event(
        "training_start",
        user_id=user_id,
        job_id=job_id,
        model_id=body.config.get("model_id"),
    )
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
