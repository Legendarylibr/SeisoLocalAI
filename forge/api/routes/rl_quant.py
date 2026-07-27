"""Adaptive RL quantization job routes."""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from forge.api.deps import get_db, get_rl_quant_orchestrator
from forge.api.http_errors import raise_forbidden
from forge.api.routes._jobs import (
    format_rl_quant_job,
    resolve_linked_training_job,
    validate_pipeline_paths,
)
from forge.api.routes._pipeline import (
    FormattedJobRoutes,
    PipelineJobResponse,
    register_formatted_job_routes,
)
from forge.api.routes._stream import spawn_background
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.rl_quant import RLQuantOrchestrator
from forge.security.audit import audit_event
from forge.security.auth import get_current_user_id
from forge.services.job_runtime import run_orchestrated_job
from forge.services.model_registry import register_export_outputs
from forge.services.user_paths import assert_rl_quant_input_paths
from seiso.bundled.config_builder import validate_stages
from seiso.rl_quant.config_builder import peek_rl_quant_input_paths
from seiso.rl_quant.presets import STAGE_ORDER, rl_quant_presets_response
from seiso.rl_quant.recommendation import (
    recommendation_evidence,
    recommendation_to_gguf_quants,
)
from seiso.security import SecurityError

router = APIRouter(prefix="/rl-quant", tags=["rl-quant"])


def _recommendation_events(result: dict[str, Any]) -> list[dict[str, str]]:
    rec = result.get("recommendation")
    if rec:
        return [{"event": "recommendation", "data": json.dumps(rec)}]
    return []


register_formatted_job_routes(
    router,
    FormattedJobRoutes(
        format_job=format_rl_quant_job,
        list_jobs=lambda db, uid: db.list_rl_quant_jobs(uid),
        get_job=lambda db, jid, uid: db.get_rl_quant_job(jid, uid),
        get_orchestrator=get_rl_quant_orchestrator,
        before_result=_recommendation_events,
    ),
)


class RLQuantStartRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    preset: str = Field(
        default="reproducible", description="reproducible | minimal | post_train"
    )
    stages: list[str] | None = None
    config_file: str | None = None
    run_name: str | None = None
    training_episodes: int | None = None
    evaluation_episodes: int | None = None
    seed: int = 13
    backend: str = Field(default="simulator", description="simulator | llama_cpp")
    training_backend: str = Field(default="stdlib", description="stdlib | pytorch")
    checkpoint_path: str | None = None
    gguf_path: str | None = None
    llama_cpp_binary: str | None = None
    gguf_export: bool = False
    moe_enabled: bool = False
    reward_weights: dict[str, float] | None = None
    link_training_job_id: str | None = None
    kernel_rl_enabled: bool = False
    kernel_live_benchmark: bool = False
    kernel_hidden_dim: int | None = None
    kernel_batch_rows: int | None = None
    kernel_benchmark_every_n_episodes: int | None = None
    auto_sweep: bool = Field(
        default=True,
        description="Grid-search key hyperparameters before the full RL quant run.",
    )
    sweep_config: str | None = Field(
        default=None,
        description="Optional sweep grid JSON/TOML under configs/ or the bundled tree.",
    )


def _prepare_rl_quant_config(
    body: RLQuantStartRequest,
    user_id: str,
    settings: ForgeSettings,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate body paths, then re-check paths after config_file/preset merge."""
    payload = dict(config or body.model_dump())
    validate_pipeline_paths(
        settings.data_dir,
        user_id,
        payload,
        config_file=body.config_file,
        path_keys=("checkpoint_path", "gguf_path"),
        llama_cpp_binary=True,
    )
    # Merge config_file/preset so tenants cannot smuggle cross-user model paths.
    try:
        merged_paths = peek_rl_quant_input_paths(payload)
        assert_rl_quant_input_paths(settings.data_dir, user_id, merged_paths)
    except SecurityError as exc:
        raise_forbidden(exc)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid RL-quant configuration: {exc}",
        ) from exc
    return payload


@router.post("/jobs", response_model=PipelineJobResponse)
async def start_rl_quant(
    body: RLQuantStartRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[RLQuantOrchestrator, Depends(get_rl_quant_orchestrator)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> PipelineJobResponse:
    job_id = str(uuid.uuid4())
    config = body.model_dump()
    if body.stages:
        validate_stages(body.stages, STAGE_ORDER)

    if body.link_training_job_id:
        await resolve_linked_training_job(
            db, user_id, body.link_training_job_id, config, path_key="checkpoint_path"
        )

    config = _prepare_rl_quant_config(body, user_id, settings, config=config)

    await db.create_rl_quant_job(user_id, config, job_id=job_id)
    orchestrator.create_job(job_id=job_id, user_id=user_id)
    payload = {**config, "user_id": user_id}

    async def _finished(job) -> None:
        rec = (job.result or {}).get("recommendation")
        evidence = recommendation_evidence(rec or {})
        # Do not advertise export quants from simulator-only evidence.
        gguf_quants = (
            recommendation_to_gguf_quants(rec or {})
            if evidence["deploy_quality_claimable"]
            else []
        )
        await db.update_rl_quant_job_status(
            job_id,
            job.status.value,
            user_id=user_id,
            output_dir=(job.result or {}).get("output_dir"),
            recommendation_path=(job.result or {}).get("recommendation_path"),
            recommendation_json=rec,
            gguf_quants=gguf_quants,
            error_text=job.error if job.status.value == "failed" else None,
        )
        exported = (job.result or {}).get("summary", {})
        gguf_path = None
        if isinstance(exported, dict):
            artifacts = exported.get("artifacts") or {}
            if isinstance(artifacts, dict):
                gguf_path = artifacts.get("exported_gguf")
        if gguf_path:
            try:
                await register_export_outputs(
                    db,
                    user_id=user_id,
                    data_dir=settings.data_dir,
                    outputs={"gguf_rl": str(gguf_path)},
                    job_id=job_id,
                )
            except Exception:
                import logging

                logging.getLogger(__name__).exception(
                    "RL quant inventory registration failed for job %s "
                    "(job remains completed)",
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
                    output_dir=(job.result or {}).get("output_dir"),
                    expected_pubkey=str((user or {}).get("nostr_pubkey") or "")
                    or None,
                )
            except Exception:
                import logging

                logging.getLogger(__name__).exception(
                    "Nostr auto-attest failed for rl-quant job %s", job_id
                )

    async def _failed(message: str) -> None:
        await db.update_rl_quant_job_status(
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
    audit_event("rl_quant_start", user_id=user_id, job_id=job_id, preset=body.preset)
    return PipelineJobResponse(job_id=job_id, status="pending")


@router.get("/presets")
async def list_presets(
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict[str, Any]:
    del user_id
    return rl_quant_presets_response()
