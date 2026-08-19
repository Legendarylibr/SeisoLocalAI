"""Training job routes."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from forge.api.deps import get_db, get_training_orchestrator
from forge.api.http_errors import raise_forbidden
from forge.api.routes._stream import (
    durable_job_events,
    job_failure_message,
    spawn_background,
)
from forge.api.schemas.training import (
    CloudGpuCredentialCreate,
    DatasetValidationRequest,
    TrainingJobResponse,
    TrainingStartRequest,
)
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.base import JobStatus
from forge.orchestrators.training import TrainingOrchestrator
from forge.security.audit import audit_event
from forge.security.auth import get_current_user_id
from forge.services.hardware import hardware_profile
from forge.services.hf_hub import search_huggingface_datasets
from forge.services.jobs import assert_job_owner
from forge.services.models import list_trainable_models, resolve_training_model_id
from forge.services.training_service import (
    cloud_credential_response,
    cloud_gpu_provider_type,
    dataset_analysis_token_matches,
    format_training_job_row,
    get_cached_dataset_analysis,
    run_dataset_analysis,
    serialize_metrics_payload,
    store_dataset_analysis_token,
)
from forge.services.user_paths import (
    assert_user_training_config,
    resolve_training_dataset_path,
)
from seiso.models.hf_env import configure_hf_hub_cache
from seiso.models.hub_quant import native_quant_training_block_reason
from seiso.models.trainable_snapshot import is_gguf_only_repo_id
from seiso.security import SecurityError, safe_join
from seiso.training.access import (
    FRONTEND_SURFACE,
    assert_surface_distributed_config,
    frontend_training_surface,
)
from seiso.training.config import DatasetFormat, TrainConfig
from seiso.training.recommendations import recommend_training_config

router = APIRouter(prefix="/training", tags=["training"])
logger = logging.getLogger(__name__)


@router.get("/surface")
async def training_surface(
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict:
    """Frontend training surface: full local config exposed; mesh/multi-node off.

    The Forge UI must keep showing proper training settings (method, quant,
    local multi-GPU DDP, hyperparams). Multi-node / mesh is Buzz-agent-only.
    """
    _ = user_id
    return frontend_training_surface()


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
        resolved_dataset = _resolve_dataset_for_user(
            dataset,
            user_id=user_id,
            settings=settings,
        )
    return recommend_training_config(
        profile,
        model_id=model_id,
        dataset=str(resolved_dataset),
        sandbox_root=Path(settings.data_dir),
        sandbox_user_id=user_id,
    )


async def _analyze_dataset_shared(
    body: DatasetValidationRequest,
    *,
    user_id: str,
    settings: ForgeSettings,
    require_valid: bool,
) -> dict:
    """Shared analyze/validate path with content-addressed result cache."""
    ds = _resolve_dataset_for_user(body.dataset, user_id=user_id, settings=settings)
    ds_fmt = DatasetFormat(body.dataset_format) if body.dataset_format else DatasetFormat.AUTO
    analysis = await asyncio.to_thread(
        run_dataset_analysis,
        ds,
        dataset_format=ds_fmt,
        sandbox_root=Path(settings.data_dir),
        sandbox_user_id=user_id,
    )
    if require_valid and not analysis.get("valid"):
        raise ValueError("No valid training samples after preprocessing")
    # Snapshot before attaching a per-request token (result cache must stay token-free).
    analysis_snapshot = dict(analysis)
    analysis["analysis_token"] = store_dataset_analysis_token(
        user_id=user_id,
        dataset=ds,
        requested_format=ds_fmt,
        resolved_format=analysis.get("resolved_format"),
        valid=bool(analysis.get("valid")),
        analysis=analysis_snapshot,
    )
    return analysis


@router.post("/analyze-dataset")
async def analyze_dataset_endpoint(
    body: DatasetValidationRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    """Research-grade dataset analysis: full-corpus schema detection and training hints."""
    try:
        return await _analyze_dataset_shared(
            body, user_id=user_id, settings=settings, require_valid=False
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/validate-dataset")
async def validate_dataset_endpoint(
    body: DatasetValidationRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    """Preflight endpoint — reuses analyze cache when the corpus was just scanned."""
    try:
        analysis = await _analyze_dataset_shared(
            body, user_id=user_id, settings=settings, require_valid=True
        )
        return {"valid": True, **analysis}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/cloud-credentials")
async def list_cloud_gpu_credentials(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> list[dict[str, Any]]:
    rows = await db.list_providers(user_id)
    return [
        cloud_credential_response(row)
        for row in rows
        if row["provider_type"] == cloud_gpu_provider_type()
    ]


@router.post("/cloud-credentials", status_code=201)
async def create_cloud_gpu_credential(
    body: CloudGpuCredentialCreate,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> dict[str, Any]:
    provider = body.provider.strip().lower()
    config = body.model_dump()
    config["provider"] = provider
    row = await db.create_provider(
        user_id,
        body.name.strip(),
        cloud_gpu_provider_type(),
        config,
    )
    audit_event(
        "cloud_gpu_credential_create",
        user_id=user_id,
        provider_id=row["id"],
        provider=provider,
    )
    return cloud_credential_response(row)


@router.delete("/cloud-credentials/{credential_id}")
async def delete_cloud_gpu_credential(
    credential_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> dict[str, str]:
    row = await db.get_provider(credential_id, user_id)
    if not row or row["provider_type"] != cloud_gpu_provider_type():
        raise HTTPException(404, "Cloud GPU credential not found")
    ok = await db.delete_provider(credential_id, user_id)
    if not ok:
        raise HTTPException(404, "Cloud GPU credential not found")
    audit_event(
        "cloud_gpu_credential_delete",
        user_id=user_id,
        provider_id=credential_id,
    )
    return {"status": "deleted"}


@router.get("/jobs")
async def list_jobs(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[TrainingOrchestrator, Depends(get_training_orchestrator)],
) -> list[dict]:
    rows = await db.list_training_jobs(user_id)
    return [format_training_job_row(row, orchestrator) for row in rows]


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
    return format_training_job_row(row, orchestrator)


@router.post("/jobs", response_model=TrainingJobResponse)
async def start_training(
    body: TrainingStartRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[TrainingOrchestrator, Depends(get_training_orchestrator)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> TrainingJobResponse:
    training_config = dict(body.config)
    install_root = Path(__file__).resolve().parents[3]
    dataset = training_config.get("dataset")
    if isinstance(dataset, str):
        training_config["dataset"] = resolve_training_dataset_path(
            settings.data_dir,
            user_id,
            dataset,
            install_root=install_root,
        )
    # Alias + resolve held-out eval the same way as train dataset.
    eval_key = (
        "slime_eval_dataset"
        if training_config.get("slime_eval_dataset")
        else "eval_dataset"
        if training_config.get("eval_dataset")
        else None
    )
    if eval_key is not None:
        eval_raw = training_config.get(eval_key)
        if isinstance(eval_raw, str):
            resolved_eval = resolve_training_dataset_path(
                settings.data_dir,
                user_id,
                eval_raw,
                install_root=install_root,
            )
            training_config["slime_eval_dataset"] = resolved_eval
            if eval_key == "eval_dataset":
                training_config.pop("eval_dataset", None)

    try:
        assert_user_training_config(settings.data_dir, user_id, training_config)
    except SecurityError as exc:
        raise_forbidden(exc)

    from forge.services.dataset_security import warn_instruction_like_dataset

    warn_instruction_like_dataset(
        training_config.get("dataset"),
        user_id=user_id,
    )

    # Forge HTTP is the frontend surface: full local training config, no mesh.
    try:
        assert_surface_distributed_config(FRONTEND_SURFACE, training_config)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    try:
        TrainConfig.model_validate(training_config)
    except Exception as exc:
        raise HTTPException(400, f"Invalid training configuration: {exc}") from exc
    cloud_credential_id = training_config.get("cloud_gpu_credential_id")
    if cloud_credential_id:
        credential = await db.get_provider(str(cloud_credential_id), user_id)
        if not credential or credential["provider_type"] != cloud_gpu_provider_type():
            raise HTTPException(400, "Cloud GPU credential not found")

    # Early dataset normalization check — fail fast with clear error *before* queuing the job
    # or downloading the base model. This fulfills "show error before training".
    dataset_for_val = training_config.get("dataset")
    ds_fmt_str = training_config.get("dataset_format", "auto")
    try:
        ds_fmt = DatasetFormat(ds_fmt_str) if ds_fmt_str else DatasetFormat.AUTO
        if not dataset_analysis_token_matches(
            body.dataset_analysis_token,
            user_id=user_id,
            dataset=dataset_for_val,
            dataset_format=ds_fmt,
        ):
            analysis = await asyncio.to_thread(
                run_dataset_analysis,
                dataset_for_val,
                dataset_format=ds_fmt,
                sandbox_root=Path(settings.data_dir),
                sandbox_user_id=user_id,
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
    except SecurityError as exc:
        raise_forbidden(exc)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    training_config = {**training_config, "model_id": resolved_model_id}
    training_config.setdefault("extra", {})
    if local_path or resolved_model_id != original_model_id:
        training_config["extra"]["resolved_model_path"] = local_path or resolved_model_id
        training_config["extra"]["original_model_id"] = original_model_id
    resolved_block = native_quant_training_block_reason(resolved_model_id)
    if resolved_block:
        raise HTTPException(400, resolved_block)

    job_id = str(uuid.uuid4())
    await db.create_training_job(user_id, training_config, body.project_id, job_id=job_id)
    orchestrator.create_job(job_id=job_id, user_id=user_id)
    extra = {**training_config.get("extra", {}), "user_id": user_id}
    if body.dataset_analysis_token:
        extra["dataset_analysis_token"] = body.dataset_analysis_token
        cached = get_cached_dataset_analysis(
            body.dataset_analysis_token,
            user_id=user_id,
            dataset=dataset_for_val,
            dataset_format=ds_fmt,
        )
        if cached is not None:
            # Pass analysis snapshot so the trainer can skip a third full scan.
            extra["cached_dataset_analysis"] = cached
    payload = {
        "config": {
            **training_config,
            # Match assert_user_training_config: full data dir + per-user scope.
            "sandbox_root": str(settings.data_dir),
            "sandbox_user_id": user_id,
            "extra": extra,
        },
        "output_dir": str(safe_join(settings.checkpoints_dir, user_id, job_id)),
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
            jid, serialize_metrics_payload(points, summary), user_id=user_id
        )

    orchestrator.set_metrics_persister(job_id, persist_metrics)

    async def _run() -> None:
        try:
            await db.update_job_status(job_id, "running", user_id=user_id)
            await orchestrator.start(job_id, payload)
            job = await orchestrator.wait_for(job_id)
            if job:
                # Never overwrite a sticky terminal cancel written by cancel_training.
                existing = await db.get_training_job(job_id, user_id)
                existing_status = str((existing or {}).get("status") or "").lower()
                if existing_status == "cancelled" and job.status.value in (
                    "completed",
                    "failed",
                ):
                    job.status = JobStatus.CANCELLED
                metrics_payload = serialize_metrics_payload(
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
                if job.status.value == "completed" and job.result.get("checkpoint_path"):
                    export_error: str | None = None
                    try:
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
                            from forge.services.memory_release import (
                                prepare_for_gpu_task,
                                release_after_task,
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
                                hub_token = resolve_hub_publish_token(settings, user_id, hub)

                            export_dir = safe_join(
                                settings.data_dir,
                                "exports",
                                user_id,
                                f"train-{job_id}",
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
                            export_job_id = f"train-export-{job_id}"
                            prep = prepare_for_gpu_task(
                                task="export",
                                job_id=export_job_id,
                                user_id=user_id,
                            )
                            try:
                                outputs = await asyncio.to_thread(
                                    auto_export_after_training,
                                    Path(job.result["checkpoint_path"]),
                                    export_dir,
                                    export_cfg,
                                    sandbox_root=settings.data_dir,
                                )
                            finally:
                                release_after_task(
                                    reason="export_on_complete",
                                    resource_token=prep.get("resource_token"),
                                    job_id=export_job_id,
                                )
                            await register_export_outputs(
                                db,
                                user_id=user_id,
                                data_dir=settings.data_dir,
                                outputs={k: str(v) for k, v in outputs.items()},
                                job_id=f"train-{job_id}",
                            )
                    except Exception as exc:
                        export_error = str(exc) or type(exc).__name__
                        logger.exception(
                            "Post-training registration/export failed for job %s",
                            job_id,
                        )
                    if export_error and body.export_on_complete:
                        # Training succeeded but requested export failed — surface it.
                        await db.update_job_status(
                            job_id,
                            "completed",
                            checkpoint_path=job.result.get("checkpoint_path"),
                            metrics=metrics_payload,
                            error_text=f"export_on_complete failed: {export_error}",
                            user_id=user_id,
                        )
        except Exception as exc:
            existing = await db.get_training_job(job_id, user_id)
            if str((existing or {}).get("status") or "").lower() == "cancelled":
                return
            error_text = job_failure_message(orchestrator, job_id, exc)
            # If the pre-start status update raised, start() never ran: fail the
            # in-memory record so streams close and the job becomes evictable.
            orchestrator.fail_unstarted_job(job_id, error_text)
            await db.update_job_status(
                job_id,
                "failed",
                error_text=error_text,
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
    if orchestrator.get_job(job_id):
        assert_job_owner(orchestrator, job_id, user_id)

    live = orchestrator.get_metrics(job_id)
    if live:
        return serialize_metrics_payload(live)

    durable_metrics = await db.list_job_events(job_id, user_id, event_types=("metric",), limit=5000)
    if durable_metrics:
        return serialize_metrics_payload([row.get("payload") or {} for row in durable_metrics])

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
    if orchestrator.get_job(job_id):
        assert_job_owner(orchestrator, job_id, user_id)

    async def event_gen():
        queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
        live_job = orchestrator.get_job(job_id)

        if not live_job:
            async for event in durable_job_events(db, job_id, user_id):
                yield event
            row = await db.get_training_job(job_id, user_id)
            if row and row.get("error_text"):
                yield {"event": "error", "data": row["error_text"]}
            status = row.get("status", "unknown") if row else "unknown"
            yield {"event": "status", "data": status}
            return

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
        try:
            while finished < 2:
                item = await queue.get()
                if item is None:
                    finished += 1
                    continue
                event, data = item
                yield {"event": event, "data": data}
        finally:
            # Client disconnects must not leak the forward tasks (or their queues).
            log_task.cancel()
            metric_task.cancel()
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
    # Allow durable-only cancel after restart / MAX_JOBS eviction (no in-memory job).
    assert_job_owner(orchestrator, job_id, user_id, allow_missing=True)
    from seiso.training.cancel import request as request_cancel

    request_cancel(job_id)
    if orchestrator.get_job(job_id):
        ok = await orchestrator.cancel(job_id)
    else:
        ok = True
    if ok:
        await db.update_job_status(job_id, "cancelled", user_id=user_id)
    return {"cancelled": ok}
