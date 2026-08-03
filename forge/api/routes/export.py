"""Export job routes."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from forge.api.deps import get_db, get_export_orchestrator, get_hub_publish_orchestrator
from forge.api.http_errors import raise_forbidden
from forge.api.routes._pipeline import PipelineJobResponse
from forge.api.routes._stream import durable_job_events, spawn_background
from forge.api.schemas.export import (
    ExportStartRequest,
    HubPrecheckRequest,
    PublishToHubRequest,
)
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.export import ExportOrchestrator
from forge.orchestrators.hub_publish import HubPublishOrchestrator
from forge.security.audit import audit_event
from forge.security.auth import get_current_user_id
from forge.services.export_publish import loads_json_field, resolve_publish_folder
from forge.services.hub_publish import (
    HubPublishRequest,
    hub_metadata_from_request,
    resolve_hub_publish_token,
)
from forge.services.job_runtime import run_orchestrated_job
from forge.services.jobs import assert_job_owner
from forge.services.publishable import (
    assert_pushable_checkpoint,
    list_publishable_models,
)
from forge.services.user_paths import assert_user_path, pick_user_download_file
from seiso.export.formats import publish_folder_to_hub
from seiso.export.model_card import HubModelMetadata
from seiso.security import SecurityError, safe_join

router = APIRouter(prefix="/export", tags=["export"])


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


def _resolve_export_hub_token(
    settings: ForgeSettings, user_id: str, hub: HubPublishRequest | None
) -> str | None:
    if hub and hub.hf_token and hub.hf_token.strip():
        return hub.hf_token.strip()
    return resolve_hub_publish_token(settings, user_id, hub)


@router.get("/profiles")
async def export_profiles(
    _user_id: Annotated[str, Depends(get_current_user_id)],
) -> list[dict]:
    from seiso.export.pipeline import profile_catalog

    return profile_catalog()


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
    token = _resolve_export_hub_token(settings, user_id, body.hub)
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
    hub_token = _resolve_export_hub_token(settings, user_id, body.hub)
    if hub_repo and not hub_token:
        raise HTTPException(
            400,
            "Hugging Face token required. Enter an API token, save one in Settings, set SEISO_HF_TOKEN, or run `hf auth login`.",
        )

    job_id = str(uuid.uuid4())
    config = body.model_dump()
    # Never persist secrets in job config_json.
    hub_cfg = config.get("hub")
    if isinstance(hub_cfg, dict) and hub_cfg.get("hf_token"):
        redacted_hub = {**hub_cfg}
        redacted_hub.pop("hf_token", None)
        config["hub"] = redacted_hub
    gguf_quants = list(body.gguf_quantizations or ["q4_k_m"])

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
        "output_dir": str(safe_join(settings.data_dir, "exports", user_id, job_id)),
        "hub_repo": hub_repo,
        "hub_token": hub_token,
        "hub_metadata": hub_metadata.__dict__ if hub_metadata else None,
        "hub_precheck": hub_precheck_dict,
    }

    async def _finished(job) -> None:
        await db.update_export_job_status(
            job_id,
            job.status.value,
            user_id=user_id,
            output_paths=job.result.get("outputs"),
            error_text=job.error if job.status.value == "failed" else None,
        )
        if job.status.value == "completed" and job.result.get("outputs"):
            from forge.services.model_registry import register_export_outputs

            try:
                await register_export_outputs(
                    db,
                    user_id=user_id,
                    data_dir=settings.data_dir,
                    outputs=job.result["outputs"],
                    job_id=job_id,
                )
            except Exception:
                logging.getLogger(__name__).exception(
                    "Export inventory registration failed for job %s "
                    "(export remains completed)",
                    job_id,
                )
        if job.status.value == "completed":
            try:
                from forge.services.nostr_settings import forge_maybe_attest

                user = await db.get_user_by_id(user_id)
                forge_maybe_attest(
                    data_dir=settings.data_dir,
                    user_id=user_id,
                    result=job.result if isinstance(job.result, dict) else None,
                    output_dir=payload.get("output_dir"),
                    expected_pubkey=str((user or {}).get("nostr_pubkey") or "")
                    or None,
                )
            except Exception:
                logging.getLogger(__name__).exception(
                    "Nostr auto-attest failed for export job %s", job_id
                )

    async def _failed(message: str) -> None:
        await db.update_export_job_status(
            job_id, "failed", user_id=user_id, error_text=message
        )

    async def _run() -> None:
        await run_orchestrated_job(
            orchestrator=orchestrator,
            job_id=job_id,
            payload=payload,
            on_finished=_finished,
            on_failed=_failed,
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

    folder, job_id, source = await resolve_publish_folder(
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
    # Persist redacted config only — never store the HF token (F4-06).
    durable_config = {
        "folder": str(folder),
        "repo_id": repo_id,
        "source_job_id": job_id,
        "source": source,
        "quantizations": meta.quantizations or None,
        "hub": {
            "repo_id": repo_id,
            "private": bool(getattr(meta, "private", False)),
        },
    }
    await db.create_hub_publish_job(user_id, durable_config, job_id=publish_job_id)
    orchestrator.create_job(job_id=publish_job_id, user_id=user_id)
    payload: dict[str, Any] = {
        "user_id": user_id,
        "folder": str(folder),
        "repo_id": repo_id,
        "token": token,
        "metadata": meta.__dict__,
        "quantizations": meta.quantizations or None,
    }

    async def _finished(job) -> None:
        await db.update_hub_publish_job_status(
            publish_job_id,
            job.status.value,
            user_id=user_id,
            result=job.result if isinstance(job.result, dict) else None,
            error_text=job.error if job.status.value == "failed" else None,
        )

    async def _failed(message: str) -> None:
        await db.update_hub_publish_job_status(
            publish_job_id, "failed", user_id=user_id, error_text=message
        )

    async def _run() -> None:
        await run_orchestrated_job(
            orchestrator=orchestrator,
            job_id=publish_job_id,
            payload=payload,
            on_finished=_finished,
            on_failed=_failed,
        )

    spawn_background(_run())
    audit_event("hf_publish_start", user_id=user_id, repo_id=repo_id, path=str(folder))
    return PipelineJobResponse(job_id=publish_job_id, status="pending")


@router.get("/publish/jobs/{job_id}/stream")
async def stream_publish_to_hub(
    job_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[
        HubPublishOrchestrator, Depends(get_hub_publish_orchestrator)
    ],
):
    if not await db.get_hub_publish_job(job_id, user_id):
        raise HTTPException(404, "Job not found")
    if orchestrator.get_job(job_id):
        assert_job_owner(orchestrator, job_id, user_id)

    async def event_gen():
        if orchestrator.get_job(job_id):
            async for line in orchestrator.stream_logs(job_id):
                yield {"event": "log", "data": line}
            j = orchestrator.get_job(job_id)
        else:
            async for event in durable_job_events(db, job_id, user_id):
                yield event
            j = None
        if not j:
            row = await db.get_hub_publish_job(job_id, user_id)
            if row and row.get("error_text"):
                yield {"event": "error", "data": row["error_text"]}
            result = loads_json_field(row.get("result_json") if row else None, {})
            if result and (not row or row.get("status") == "completed"):
                yield {"event": "result", "data": json.dumps(result, default=str)}
            return
        if j.error:
            yield {"event": "error", "data": j.error}
        if j.result and j.status.value == "completed":
            yield {"event": "result", "data": json.dumps(j.result, default=str)}

    return EventSourceResponse(event_gen())


@router.post("/publish")
async def publish_to_hub(
    body: PublishToHubRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict[str, str]:
    """Publish synchronously.

    Deprecated for UI/large GGUF: prefer POST /export/publish/jobs.
    """
    token = resolve_hub_publish_token(settings, user_id, body.hub)
    if not token:
        raise HTTPException(
            400,
            "Hugging Face token required. Enter an API token, save one in Settings, set SEISO_HF_TOKEN, or run `hf auth login`.",
        )

    meta = hub_metadata_from_request(body.hub)
    meta.validate()
    repo_id = meta.repo_id

    folder, job_id, source = await resolve_publish_folder(
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
            # Re-precheck at push time (token/ownership can change after route check).
            skip_precheck=False,
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

    key = (key or "").strip()
    if not key:
        raise HTTPException(400, "Output key is required")
    outputs = loads_json_field(job.get("output_paths_json") or "{}", {})
    target_raw = outputs.get(key)
    if not target_raw:
        # Case-insensitive exact match only — never substring (avoids key="" / "a"
        # resolving to an unintended artifact).
        lowered = key.lower()
        for k, v in outputs.items():
            if str(k).lower() == lowered:
                target_raw = v
                break
    if not target_raw:
        raise HTTPException(404, f"Output key {key!r} not found")

    try:
        path = assert_user_path(settings.data_dir, user_id, target_raw)
    except SecurityError as exc:
        raise_forbidden(exc)

    if path.is_dir():
        try:
            path = pick_user_download_file(
                settings.data_dir,
                user_id,
                path,
                pattern="*.gguf",
            )
        except SecurityError as exc:
            raise_forbidden(exc)
    else:
        try:
            path = assert_user_path(settings.data_dir, user_id, path)
        except SecurityError as exc:
            raise_forbidden(exc)

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
    if orchestrator.get_job(job_id):
        assert_job_owner(orchestrator, job_id, user_id)

    async def event_gen():
        if orchestrator.get_job(job_id):
            async for line in orchestrator.stream_logs(job_id):
                yield {"event": "log", "data": line}
            j = orchestrator.get_job(job_id)
        else:
            async for event in durable_job_events(db, job_id, user_id):
                yield event
            j = None
        if not j:
            row = await db.get_export_job(job_id, user_id)
            if row and row.get("error_text"):
                yield {"event": "error", "data": row["error_text"]}
            outputs = loads_json_field(row.get("output_paths_json") if row else None, {})
            if outputs and (not row or row.get("status") == "completed"):
                yield {"event": "result", "data": json.dumps({"outputs": outputs})}
            return
        if j.error:
            yield {"event": "error", "data": j.error}
        if j.result and j.status.value == "completed":
            yield {"event": "result", "data": json.dumps(j.result, default=str)}

    return EventSourceResponse(event_gen())
