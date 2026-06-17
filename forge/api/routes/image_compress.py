"""Stable Diffusion image compression pipeline job routes."""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from forge.api.deps import get_db, get_image_compress_orchestrator
from forge.api.routes._stream import job_failure_message, job_log_event_gen, spawn_background
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.image_compress import ImageCompressOrchestrator
from forge.security.audit import audit_event
from forge.security.auth import get_current_user_id
from forge.services.jobs import assert_job_owner
from forge.services.model_registry import register_export_outputs
from forge.services.user_paths import assert_user_path
from seiso.image_compress.config_builder import PRESETS, STAGE_ORDER
from seiso.security import SecurityError

router = APIRouter(prefix="/image-compress", tags=["image-compress"])


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


class ImageCompressStartRequest(BaseModel):
    preset: str = Field(
        default="smoke",
        description="smoke | full | distill_only | prune_recover | quantize",
    )
    stages: list[str] | None = None
    base_model: str = "runwayml/stable-diffusion-v1-5"
    model_dir: str | None = None
    data_path: str | None = None
    steps: int | None = None
    clip_distill_steps: int | None = None
    cfg_distill_steps: int | None = None
    finetune_steps: int | None = None
    prune_ratio: float | None = None
    text_encoder_prune_ratio: float | None = None
    vae_prune_ratio: float | None = None
    int8_calibration_samples: int | None = None
    eval_samples: int | None = None
    eval_inference_steps: int | None = None
    export_model_name: str = "seiso-sd-compressed"


class ImageCompressJobResponse(BaseModel):
    job_id: str
    status: str


@router.get("/jobs")
async def list_image_compress_jobs(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> list[dict]:
    return [_format_job(j) for j in await db.list_image_compress_jobs(user_id)]


@router.get("/jobs/{job_id}")
async def get_image_compress_job(
    job_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> dict:
    job = await db.get_image_compress_job(job_id, user_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return _format_job(job)


@router.post("/jobs", response_model=ImageCompressJobResponse)
async def start_image_compress(
    body: ImageCompressStartRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[ImageCompressOrchestrator, Depends(get_image_compress_orchestrator)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> ImageCompressJobResponse:
    job_id = str(uuid.uuid4())
    config = body.model_dump()

    if config.get("model_dir"):
        try:
            assert_user_path(settings.data_dir, user_id, config["model_dir"])
        except SecurityError as exc:
            raise HTTPException(403, str(exc)) from exc

    if config.get("data_path"):
        try:
            assert_user_path(settings.data_dir, user_id, config["data_path"])
        except SecurityError as exc:
            raise HTTPException(403, str(exc)) from exc

    await db.create_image_compress_job(user_id, config, job_id=job_id)
    orchestrator.create_job(job_id=job_id, user_id=user_id)
    payload = {**config, "user_id": user_id}

    async def _run() -> None:
        try:
            await orchestrator.start(job_id, payload)
            job = await orchestrator.wait_for(job_id)
            if job:
                result = job.result or {}
                stage_results = dict(result.get("stage_results") or {})
                if result.get("provenance_path"):
                    stage_results["provenance_path"] = result["provenance_path"]
                if result.get("checksum_manifest"):
                    stage_results["checksum_manifest"] = result["checksum_manifest"]
                await db.update_image_compress_job_status(
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
                    await register_export_outputs(
                        db,
                        user_id=user_id,
                        data_dir=settings.data_dir,
                        outputs={"sd_compressed": str(model_dir)},
                        job_id=job_id,
                    )
        except Exception as exc:
            await db.update_image_compress_job_status(
                job_id,
                "failed",
                error_text=job_failure_message(orchestrator, job_id, exc),
            )

    spawn_background(_run())
    audit_event("image_compress_start", user_id=user_id, job_id=job_id, preset=body.preset)
    return ImageCompressJobResponse(job_id=job_id, status="pending")


@router.get("/jobs/{job_id}/stream")
async def stream_image_compress(
    job_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[ImageCompressOrchestrator, Depends(get_image_compress_orchestrator)],
):
    if not await db.get_image_compress_job(job_id, user_id):
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
            "baseline": "Generate reference images for quality comparison",
            "distill_progressive": "Progressive step-halving UNet distillation",
            "distill_clip": "CLIP text encoder distillation",
            "distill_cfg": "Classifier-free guidance distillation",
            "evaluate_distilled": "CLIP/LPIPS/SSIM vs baseline",
            "prune": "Structured pruning (text encoder, VAE, UNet)",
            "evaluate_pruned": "Quality check after pruning",
            "finetune": "Recovery fine-tune after pruning",
            "evaluate_finetuned": "Quality check after fine-tuning",
            "quantize": "FP16 + INT8 quantisation",
            "evaluate_quantized": "Final quality + size metrics",
            "optimize": "ToMe + torch.compile runtime helpers",
            "export_onnx": "ONNX export (requires pip install seiso[image-compress-onnx])",
            "export_shard": "Shard safetensors for deployment",
            "lora_test": "LoRA compatibility check",
            "report": "Aggregate stage metrics into full report",
        },
    }
