"""Code Llama compression pipeline job routes."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from forge.api.deps import get_compress_orchestrator, get_db
from forge.api.routes._stream import job_log_event_gen
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.compress import CompressOrchestrator
from forge.security.audit import audit_event
from forge.security.auth import get_current_user_id
from forge.services.jobs import assert_job_owner
from forge.services.model_registry import register_export_outputs
from forge.services.user_paths import assert_user_config_file, assert_user_path
from seiso.compress.config_builder import PRESETS, STAGE_ORDER
from seiso.security import SecurityError

router = APIRouter(prefix="/compress", tags=["compress"])


def _format_job(row: dict) -> dict:
    out = dict(row)
    try:
        out["stages"] = json.loads(row.get("stages_json") or "[]")
    except json.JSONDecodeError:
        out["stages"] = []
    try:
        out["stage_results"] = json.loads(row.get("stage_results_json") or "{}")
    except json.JSONDecodeError:
        out["stage_results"] = {}
    return out


class CompressStartRequest(BaseModel):
    preset: str = Field(
        default="smoke",
        description="smoke | full | distill_only | prune_recover | quantize",
    )
    stages: list[str] | None = None
    config_file: str | None = None
    teacher_model: str = "codellama/CodeLlama-13b-hf"
    student_model: str = "codellama/CodeLlama-7b-hf"
    model_dir: str | None = None
    distill_steps: int | None = None
    finetune_steps: int | None = None
    prune_ratio: float | None = None
    prune_method: str = "magnitude"
    seed: int = 42
    deterministic: bool = True
    export_model_name: str = "seiso-compressed"
    calibration_samples: int | None = None
    max_train_samples: int | None = None
    link_training_job_id: str | None = None


class CompressJobResponse(BaseModel):
    job_id: str
    status: str


@router.get("/jobs")
async def list_compress_jobs(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> list[dict]:
    return [_format_job(j) for j in await db.list_compress_jobs(user_id)]


@router.get("/jobs/{job_id}")
async def get_compress_job(
    job_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> dict:
    job = await db.get_compress_job(job_id, user_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return _format_job(job)


@router.post("/jobs", response_model=CompressJobResponse)
async def start_compress(
    body: CompressStartRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[CompressOrchestrator, Depends(get_compress_orchestrator)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> CompressJobResponse:
    job_id = str(uuid.uuid4())
    config = body.model_dump()

    if body.link_training_job_id:
        train_job = await db.get_training_job(body.link_training_job_id, user_id)
        if not train_job:
            raise HTTPException(404, "Linked training job not found")
        if train_job.get("checkpoint_path"):
            config["model_dir"] = train_job["checkpoint_path"]
            if config.get("preset") == "smoke":
                config["preset"] = "prune_recover"

    try:
        if body.config_file:
            assert_user_config_file(settings.data_dir, user_id, body.config_file)
        if config.get("model_dir"):
            assert_user_path(settings.data_dir, user_id, config["model_dir"])
    except SecurityError as exc:
        raise HTTPException(403, str(exc)) from exc

    await db.create_compress_job(user_id, config, job_id=job_id)
    orchestrator.create_job(job_id=job_id, user_id=user_id)
    payload = {**config, "user_id": user_id}

    async def _run() -> None:
        try:
            await orchestrator.start(job_id, payload)
            job = await orchestrator.wait_for(job_id)
            if job:
                result = job.result or {}
                await db.update_compress_job_status(
                    job_id,
                    job.status.value,
                    output_dir=result.get("output_root"),
                    run_dir=result.get("run_dir"),
                    model_dir=result.get("model_dir"),
                    stages=result.get("stages"),
                    stage_results=result.get("stage_results"),
                )
                if model_dir := result.get("model_dir"):
                    await register_export_outputs(
                        db,
                        user_id=user_id,
                        data_dir=settings.data_dir,
                        outputs={"compressed": str(model_dir)},
                        job_id=job_id,
                    )
        except Exception:
            await db.update_compress_job_status(job_id, "failed")
            raise

    asyncio.create_task(_run())
    audit_event("compress_start", user_id=user_id, job_id=job_id, preset=body.preset)
    return CompressJobResponse(job_id=job_id, status="pending")


@router.get("/jobs/{job_id}/stream")
async def stream_compress(
    job_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[CompressOrchestrator, Depends(get_compress_orchestrator)],
):
    if not await db.get_compress_job(job_id, user_id):
        raise HTTPException(404, "Job not found")
    assert_job_owner(orchestrator, job_id, user_id)

    return EventSourceResponse(job_log_event_gen(orchestrator, job_id))


@router.get("/presets")
async def list_presets(
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict[str, Any]:
    return {
        "presets": [
            {
                "id": name,
                "label": name.replace("_", " ").title(),
                "stages": preset.get("stages", []),
            }
            for name, preset in PRESETS.items()
        ],
        "stages": list(STAGE_ORDER),
        "help": {
            "distill": "Teacher → student KL distillation",
            "prune": "Shape-preserving MLP neuron masking",
            "finetune": "Post-prune recovery fine-tuning",
            "evaluate": "Perplexity + speed smoke check",
            "export": "vLLM/Docker/GGUF helper scripts",
            "quantize_gptq": "GPTQ 4-bit (requires pip install seiso[compress-quant])",
            "quantize_awq": "AWQ 4-bit (requires pip install seiso[compress-quant])",
        },
    }
