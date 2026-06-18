"""Adaptive RL quantization job routes."""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from forge.api.deps import get_db, get_rl_quant_orchestrator
from forge.api.routes._jobs import format_rl_quant_job, resolve_linked_training_job
from forge.api.routes._pipeline import FormattedJobRoutes, PipelineJobResponse, register_formatted_job_routes
from forge.api.routes._stream import job_failure_message, spawn_background
from forge.api.routes.rl_quant_presets import rl_quant_presets_response
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.rl_quant import RLQuantOrchestrator
from forge.security.audit import audit_event
from forge.security.auth import get_current_user_id
from forge.services.model_registry import register_export_outputs
from forge.services.user_paths import (
    assert_llama_cpp_binary,
    assert_user_config_file,
    assert_user_path,
)
from seiso.rl_quant.recommendation import recommendation_to_gguf_quants
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
    preset: str = Field(default="reproducible", description="reproducible | minimal | post_train")
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

    if body.link_training_job_id:
        await resolve_linked_training_job(
            db, user_id, body.link_training_job_id, config, path_key="checkpoint_path"
        )

    try:
        if body.config_file:
            assert_user_config_file(settings.data_dir, user_id, body.config_file)
        for path_key in ("checkpoint_path", "gguf_path"):
            if config.get(path_key):
                assert_user_path(settings.data_dir, user_id, config[path_key])
        if config.get("llama_cpp_binary"):
            assert_llama_cpp_binary(config["llama_cpp_binary"])
    except SecurityError as exc:
        raise HTTPException(403, str(exc)) from exc

    await db.create_rl_quant_job(user_id, config, job_id=job_id)
    orchestrator.create_job(job_id=job_id, user_id=user_id)
    payload = {**config, "user_id": user_id}

    async def _run() -> None:
        try:
            await orchestrator.start(job_id, payload)
            job = await orchestrator.wait_for(job_id)
            if job:
                rec = (job.result or {}).get("recommendation")
                gguf_quants = recommendation_to_gguf_quants(rec or {})
                await db.update_rl_quant_job_status(
                    job_id,
                    job.status.value,
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
                    await register_export_outputs(
                        db,
                        user_id=user_id,
                        data_dir=settings.data_dir,
                        outputs={"gguf_rl": str(gguf_path)},
                        job_id=job_id,
                    )
        except Exception as exc:
            await db.update_rl_quant_job_status(
                job_id,
                "failed",
                error_text=job_failure_message(orchestrator, job_id, exc),
            )

    spawn_background(_run())
    audit_event("rl_quant_start", user_id=user_id, job_id=job_id, preset=body.preset)
    return PipelineJobResponse(job_id=job_id, status="pending")


@router.get("/presets")
async def list_presets(
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict[str, Any]:
    return rl_quant_presets_response()
