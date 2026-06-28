"""Export job routes."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from forge.api.deps import get_db, get_export_orchestrator, get_hub_publish_orchestrator
from forge.api.http_errors import raise_forbidden
from forge.api.routes._pipeline import PipelineJobResponse
from forge.api.routes._stream import job_failure_message, spawn_background
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.export import ExportOrchestrator
from forge.orchestrators.hub_publish import HubPublishOrchestrator
from forge.security.audit import audit_event
from forge.security.auth import get_current_user_id
from forge.services.hub_publish import (
    HubPublishRequest,
    hub_metadata_from_request,
    resolve_hub_publish_token,
)
from forge.services.jobs import assert_job_owner
from forge.services.publishable import (
    assert_pushable_checkpoint,
    assert_pushable_model,
    assert_pushable_path,
    list_publishable_models,
)
from forge.services.user_paths import assert_user_path
from seiso.export.formats import publish_folder_to_hub
from seiso.export.model_card import HubModelMetadata
from seiso.security import SecurityError

router = APIRouter(prefix="/export", tags=["export"])


class ExportStartRequest(BaseModel):
    checkpoint: str
    formats: list[str] = Field(default_factory=lambda: ["merged"])
    profile: str | None = Field(
        default=None,
        description="Export profile: lora_adapter, lora_bundle, full_finetune, full_bundle, inference, gguf_only, hub_ready",
    )
    gguf_quantizations: list[str] = Field(default_factory=lambda: ["q4_k_m"])
    hub: HubPublishRequest | None = None
    hub_repo: str | None = Field(
        default=None, description="Deprecated — use hub.username + hub.model_name"
    )
    rl_quant_job_id: str | None = Field(
        default=None,
        description="Apply GGUF quants from a completed RL quant recommendation job",
    )


class PublishToHubRequest(BaseModel):
    model_id: str | None = None
    output_path: str | None = None
    export_job_id: str | None = None
    hub: HubPublishRequest


async def _resolve_publish_folder(
    body: PublishToHubRequest,
    *,
    user_id: str,
    db: Database,
    data_dir: Path,
) -> tuple[Path, str | None, str]:
    """Return (folder, job_id, source) for a publish request."""
    if body.model_id:
        model = await assert_pushable_model(db, model_id=body.model_id, user_id=user_id)
        folder = Path(model["path"])
        meta_raw = json.loads(model.get("metadata_json") or "{}")
        job_id = meta_raw.get("job_id")
        source = model.get("source") or "export"
    elif body.export_job_id:
        job = await db.get_export_job(body.export_job_id, user_id)
        if not job or job.get("status") != "completed":
            raise HTTPException(400, "Export job not found or not completed")
        outputs = json.loads(job.get("output_paths_json") or "{}")
        if not outputs:
            raise HTTPException(400, "Export job has no outputs")
        preferred = next((v for k, v in outputs.items() if "gguf" in k.lower()), None)
        if not preferred:
            preferred = outputs.get("merged") or next(iter(outputs.values()))
        folder = Path(preferred)
        job_id = body.export_job_id
        source = "export"
    elif body.output_path:
        try:
            folder = await assert_pushable_path(
                db, data_dir=data_dir, user_id=user_id, target=body.output_path
            )
        except (SecurityError, ValueError) as exc:
            raise HTTPException(
                403 if isinstance(exc, SecurityError) else 400, str(exc)
            ) from exc
        job_id = None
        source = "export"
    else:
        raise HTTPException(400, "Provide model_id, export_job_id, or output_path")

    if folder.is_file():
        folder = folder.parent
    return folder, job_id, source


def _resolve_hub_repo(
    body: ExportStartRequest,
) -> tuple[str | None, HubModelMetadata | None]:
    if body.hub:
        meta = hub_metadata_from_request(body.hub)
        meta.validate()
        return meta.repo_id, meta
    if body.hub_repo:
        return body.hub_repo.strip(), None
    return None, None


@router.get("/profiles")
async def export_profiles() -> list[dict]:
    from seiso.export.pipeline import profile_catalog

    return profile_catalog()


class HubPrecheckRequest(BaseModel):
    hub: HubPublishRequest
    formats: list[str] = Field(default_factory=lambda: ["merged"])
    profile: str | None = None
    gguf_quantizations: list[str] = Field(default_factory=lambda: ["q4_k_m"])


@router.post("/precheck")
async def precheck_hub_export_route(
    body: HubPrecheckRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    """Validate Hub repo availability and model card before starting export."""
    from seiso.export.hub_precheck import precheck_hub_export

    meta = hub_metadata_from_request(body.hub)
    meta.validate()
    if body.profile or "gguf" in [f.lower() for f in body.formats]:
        meta.quantizations = list(body.gguf_quantizations)
    token = resolve_hub_publish_token(settings, user_id, body.hub)
    if not token:
        raise HTTPException(
            400,
            "Hugging Face token required. Enter an API token, save one in Settings, set SEISO_HF_TOKEN, or run `hf auth login`.",
        )
    result = precheck_hub_export(
        repo_id=meta.repo_id,
        token=token,
        metadata=meta,
        formats=body.formats,
    )
    return result.to_dict()


@router.get("/publishable")
async def publishable_outputs(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> list[dict]:
    return await list_publishable_models(db, user_id)


@router.get("/jobs")
async def list_export_jobs(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> list[dict]:
    return await db.list_export_jobs(user_id)


@router.post("/jobs", response_model=PipelineJobResponse)
async def start_export(
    body: ExportStartRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[ExportOrchestrator, Depends(get_export_orchestrator)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> PipelineJobResponse:
    try:
        await assert_pushable_checkpoint(
            db, data_dir=settings.data_dir, user_id=user_id, checkpoint=body.checkpoint
        )
    except (SecurityError, ValueError) as exc:
        raise HTTPException(
            403 if isinstance(exc, SecurityError) else 400, str(exc)
        ) from exc

    hub_repo, hub_metadata = _resolve_hub_repo(body)
    hub_token = resolve_hub_publish_token(settings, user_id, body.hub)
    if hub_repo and not hub_token:
        raise HTTPException(
            400,
            "Hugging Face token required. Enter an API token, save one in Settings, set SEISO_HF_TOKEN, or run `hf auth login`.",
        )

    job_id = str(uuid.uuid4())
    config = body.model_dump()
    gguf_quants = list(body.gguf_quantizations)

    if body.rl_quant_job_id:
        rl_job = await db.get_rl_quant_job(body.rl_quant_job_id, user_id)
        if not rl_job:
            raise HTTPException(404, "RL quant job not found")
        if rl_job.get("status") != "completed":
            raise HTTPException(400, "RL quant job is not completed")
        stored = rl_job.get("gguf_quants_json") or "[]"
        parsed = json.loads(stored)
        if parsed:
            gguf_quants = parsed
        config["rl_quant_job_id"] = body.rl_quant_job_id

    hub_precheck_dict: dict[str, Any] | None = None
    if hub_repo and hub_metadata:
        from seiso.export.hub_precheck import (
            assert_hub_precheck_ok,
            precheck_hub_export,
        )
        from seiso.export.profiles import resolve_formats

        resolved_formats = resolve_formats(
            formats=body.formats if not body.profile else None,
            profile=body.profile,
        )
        if any(f.value == "gguf" for f in resolved_formats):
            hub_metadata.quantizations = gguf_quants
        precheck = precheck_hub_export(
            repo_id=hub_repo,
            token=hub_token or "",
            metadata=hub_metadata,
            formats=[f.value for f in resolved_formats],
        )
        if not precheck.ok:
            try:
                assert_hub_precheck_ok(precheck)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        hub_precheck_dict = precheck.to_dict()

    await db.create_export_job(user_id, config, job_id=job_id)
    orchestrator.create_job(job_id=job_id, user_id=user_id)
    payload: dict[str, Any] = {
        **body.model_dump(),
        "gguf_quantizations": gguf_quants,
        "user_id": user_id,
        "output_dir": str(settings.data_dir / "exports" / user_id / job_id),
        "hub_repo": hub_repo,
        "hub_token": hub_token,
        "hub_metadata": hub_metadata.__dict__ if hub_metadata else None,
        "hub_precheck": hub_precheck_dict,
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
                    error_text=job.error if job.status.value == "failed" else None,
                )
                if job.status.value == "completed" and job.result.get("outputs"):
                    from forge.services.model_registry import register_export_outputs

                    await register_export_outputs(
                        db,
                        user_id=user_id,
                        data_dir=settings.data_dir,
                        outputs=job.result["outputs"],
                        job_id=job_id,
                    )
        except Exception as exc:
            await db.update_export_job_status(
                job_id,
                "failed",
                error_text=job_failure_message(orchestrator, job_id, exc),
            )

    spawn_background(_run())
    audit_event("export_start", user_id=user_id, job_id=job_id, formats=body.formats)
    return PipelineJobResponse(job_id=job_id, status="pending")


@router.post("/publish/jobs", response_model=PipelineJobResponse)
async def start_publish_to_hub(
    body: PublishToHubRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[
        HubPublishOrchestrator, Depends(get_hub_publish_orchestrator)
    ],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> PipelineJobResponse:
    """Start a background Hugging Face publish job (required for multi-GB GGUF uploads)."""
    token = resolve_hub_publish_token(settings, user_id, body.hub)
    if not token:
        raise HTTPException(
            400,
            "Hugging Face token required. Enter an API token, save one in Settings, set SEISO_HF_TOKEN, or run `hf auth login`.",
        )

    meta = hub_metadata_from_request(body.hub)
    meta.validate()
    repo_id = meta.repo_id

    folder, job_id, source = await _resolve_publish_folder(
        body,
        user_id=user_id,
        db=db,
        data_dir=settings.data_dir,
    )
    meta.seiso_job_id = job_id
    meta.seiso_source = source

    from seiso.export.hub_precheck import assert_hub_precheck_ok, precheck_hub_export

    precheck = precheck_hub_export(
        repo_id=repo_id,
        token=token,
        metadata=meta,
        formats=meta.export_formats or None,
    )
    if not precheck.ok:
        try:
            assert_hub_precheck_ok(precheck)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    publish_job_id = str(uuid.uuid4())
    orchestrator.create_job(job_id=publish_job_id, user_id=user_id)
    payload: dict[str, Any] = {
        "user_id": user_id,
        "folder": str(folder),
        "repo_id": repo_id,
        "token": token,
        "metadata": meta.__dict__,
        "quantizations": meta.quantizations or None,
    }

    async def _run() -> None:
        try:
            await orchestrator.start(publish_job_id, payload)
            await orchestrator.wait_for(publish_job_id)
        except Exception as exc:
            job = orchestrator.get_job(publish_job_id)
            if job and job.status.value != "failed":
                orchestrator._emit_log(publish_job_id, f"ERROR: {exc}")

    spawn_background(_run())
    audit_event("hf_publish_start", user_id=user_id, repo_id=repo_id, path=str(folder))
    return PipelineJobResponse(job_id=publish_job_id, status="pending")


@router.get("/publish/jobs/{job_id}/stream")
async def stream_publish_to_hub(
    job_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    orchestrator: Annotated[
        HubPublishOrchestrator, Depends(get_hub_publish_orchestrator)
    ],
):
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


@router.post("/publish")
async def publish_to_hub(
    body: PublishToHubRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict[str, str]:
    """Publish synchronously — prefer POST /export/publish/jobs for large GGUF files."""
    token = resolve_hub_publish_token(settings, user_id, body.hub)
    if not token:
        raise HTTPException(
            400,
            "Hugging Face token required. Enter an API token, save one in Settings, set SEISO_HF_TOKEN, or run `hf auth login`.",
        )

    meta = hub_metadata_from_request(body.hub)
    meta.validate()
    repo_id = meta.repo_id

    folder, job_id, source = await _resolve_publish_folder(
        body,
        user_id=user_id,
        db=db,
        data_dir=settings.data_dir,
    )
    meta.seiso_job_id = job_id
    meta.seiso_source = source

    from seiso.export.hub_precheck import assert_hub_precheck_ok, precheck_hub_export

    precheck = precheck_hub_export(
        repo_id=repo_id,
        token=token,
        metadata=meta,
        formats=meta.export_formats or None,
    )
    if not precheck.ok:
        try:
            assert_hub_precheck_ok(precheck)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    logs: list[str] = []

    def on_log(msg: str) -> None:
        logs.append(msg)

    try:
        publish_folder_to_hub(
            folder,
            repo_id=repo_id,
            token=token,
            metadata=meta,
            on_log=on_log,
            skip_precheck=True,
            data_dir=settings.data_dir,
        )
    except Exception as exc:
        raise HTTPException(500, f"Hugging Face upload failed: {exc}") from exc

    audit_event("hf_publish", user_id=user_id, repo_id=repo_id, path=str(folder))
    return {"repo_id": repo_id, "path": str(folder), "log": "\n".join(logs)}


@router.get("/outputs/{job_id}/download")
async def download_export_output(
    job_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
    key: str = "gguf",
):
    """Download a GGUF or other export artifact from a completed export job."""
    job = await db.get_export_job(job_id, user_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(404, "Export job not found or not completed")

    outputs = json.loads(job.get("output_paths_json") or "{}")
    target_raw = outputs.get(key)
    if not target_raw:
        for k, v in outputs.items():
            if key.lower() in k.lower():
                target_raw = v
                break
    if not target_raw:
        raise HTTPException(404, f"Output key {key!r} not found")

    try:
        path = assert_user_path(settings.data_dir, user_id, target_raw)
    except SecurityError as exc:
        raise_forbidden(exc)

    if path.is_dir():
        ggufs = sorted(path.glob("*.gguf"))
        if not ggufs:
            raise HTTPException(404, "No GGUF file in output directory")
        path = ggufs[0]

    if not path.is_file():
        raise HTTPException(404, "File not found")

    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


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
