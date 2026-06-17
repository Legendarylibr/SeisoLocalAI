"""Adaptive RL quantization job routes."""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from forge.api.deps import get_db, get_rl_quant_orchestrator
from forge.api.routes._stream import job_log_event_gen, spawn_background
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.rl_quant import RLQuantOrchestrator
from forge.security.audit import audit_event
from forge.security.auth import get_current_user_id
from forge.services.jobs import assert_job_owner
from forge.services.model_registry import register_export_outputs
from forge.services.user_paths import assert_user_config_file, assert_user_path
from seiso.rl_quant.recommendation import recommendation_to_gguf_quants
from seiso.security import SecurityError

router = APIRouter(prefix="/rl-quant", tags=["rl-quant"])


def _format_job(row: dict) -> dict:
    out = dict(row)
    try:
        out["gguf_quants"] = json.loads(row.get("gguf_quants_json") or "[]")
    except json.JSONDecodeError:
        out["gguf_quants"] = []
    try:
        out["recommendation"] = json.loads(row.get("recommendation_json") or "{}")
    except json.JSONDecodeError:
        out["recommendation"] = {}
    return out


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


class RLQuantJobResponse(BaseModel):
    job_id: str
    status: str


@router.get("/jobs")
async def list_rl_quant_jobs(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> list[dict]:
    return [_format_job(j) for j in await db.list_rl_quant_jobs(user_id)]


@router.get("/jobs/{job_id}")
async def get_rl_quant_job(
    job_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> dict:
    job = await db.get_rl_quant_job(job_id, user_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return _format_job(job)


@router.post("/jobs", response_model=RLQuantJobResponse)
async def start_rl_quant(
    body: RLQuantStartRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[RLQuantOrchestrator, Depends(get_rl_quant_orchestrator)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> RLQuantJobResponse:
    job_id = str(uuid.uuid4())
    config = body.model_dump()

    if body.link_training_job_id:
        train_job = await db.get_training_job(body.link_training_job_id, user_id)
        if not train_job:
            raise HTTPException(404, "Linked training job not found")
        if train_job.get("checkpoint_path"):
            config["checkpoint_path"] = train_job["checkpoint_path"]

    try:
        if body.config_file:
            assert_user_config_file(settings.data_dir, user_id, body.config_file)
        for path_key in ("checkpoint_path", "gguf_path"):
            if config.get(path_key):
                assert_user_path(settings.data_dir, user_id, config[path_key])
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
        except Exception:
            await db.update_rl_quant_job_status(job_id, "failed")

    spawn_background(_run())
    audit_event("rl_quant_start", user_id=user_id, job_id=job_id, preset=body.preset)
    return RLQuantJobResponse(job_id=job_id, status="pending")


@router.get("/jobs/{job_id}/stream")
async def stream_rl_quant(
    job_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[RLQuantOrchestrator, Depends(get_rl_quant_orchestrator)],
):
    if not await db.get_rl_quant_job(job_id, user_id):
        raise HTTPException(404, "Job not found")
    assert_job_owner(orchestrator, job_id, user_id)

    def _recommendation_events(result: dict) -> list[dict[str, str]]:
        rec = result.get("recommendation")
        if rec:
            return [{"event": "recommendation", "data": json.dumps(rec)}]
        return []

    return EventSourceResponse(job_log_event_gen(orchestrator, job_id, before_result=_recommendation_events))


@router.get("/presets")
async def list_presets(
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict[str, Any]:
    return {
        "presets": [
            {
                "id": "reproducible",
                "label": "Reproducible research (simulator)",
                "backend": "simulator",
                "training_backend": "stdlib",
            },
            {
                "id": "minimal",
                "label": "Fast smoke (256 episodes)",
                "backend": "simulator",
                "training_backend": "stdlib",
            },
            {
                "id": "post_train",
                "label": "Post fine-tune RL (continuous, router)",
                "backend": "simulator",
                "training_backend": "stdlib",
            },
        ],
        "reward_weights_help": {
            "alpha_latency": "Latency penalty weight",
            "beta_throughput": "Throughput reward weight",
            "gamma_perplexity": "Quality / perplexity weight",
            "delta_memory": "Memory footprint weight",
            "epsilon_instability": "Instability probe penalty",
        },
    }
