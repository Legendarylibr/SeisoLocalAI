"""Stable Diffusion image compression pipeline job routes."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from forge.api.deps import get_image_compress_orchestrator
from forge.api.routes._pipeline import StagePipelineRouterConfig, build_stage_pipeline_router
from forge.config import ForgeSettings
from forge.db.store import Database
from forge.services.user_paths import assert_user_path
from seiso.image_compress.config_builder import PRESETS, STAGE_ORDER, get_image_base_model_default
from seiso.security import SecurityError

STAGE_HELP = {
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
}


class ImageCompressStartRequest(BaseModel):
    preset: str = Field(
        default="smoke",
        description="smoke | full | distill_only | prune_recover | quantize",
    )
    stages: list[str] | None = None
    base_model: str | None = None
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


async def _prepare_image_compress_config(
    body: BaseModel,
    db: Database,
    user_id: str,
    settings: ForgeSettings,
) -> dict[str, Any]:
    del db  # image compress does not link training jobs yet
    req = body
    assert isinstance(req, ImageCompressStartRequest)
    config = req.model_dump()
    if not config.get("base_model"):
        config["base_model"] = get_image_base_model_default()

    try:
        if config.get("model_dir"):
            assert_user_path(settings.data_dir, user_id, config["model_dir"])
        if config.get("data_path"):
            assert_user_path(settings.data_dir, user_id, config["data_path"])
    except SecurityError as exc:
        raise HTTPException(403, str(exc)) from exc

    return config


def _enrich_image_compress_stage_results(result: dict[str, Any]) -> dict[str, Any]:
    stage_results = dict(result.get("stage_results") or {})
    if result.get("provenance_path"):
        stage_results["provenance_path"] = result["provenance_path"]
    if result.get("checksum_manifest"):
        stage_results["checksum_manifest"] = result["checksum_manifest"]
    return stage_results


router = build_stage_pipeline_router(
    StagePipelineRouterConfig(
        prefix="/image-compress",
        tags=("image-compress",),
        audit_event_name="image_compress_start",
        export_registry_key="sd_compressed",
        presets=PRESETS,
        stage_order=STAGE_ORDER,
        stage_help=STAGE_HELP,
        model_defaults={"base_model": get_image_base_model_default()},
        start_request_model=ImageCompressStartRequest,
        get_orchestrator=get_image_compress_orchestrator,
        list_jobs=lambda db, uid: db.list_image_compress_jobs(uid),
        get_job=lambda db, jid, uid: db.get_image_compress_job(jid, uid),
        create_job=lambda db, uid, cfg, jid: db.create_image_compress_job(uid, cfg, job_id=jid),
        update_status=lambda db, jid, status, **kw: db.update_image_compress_job_status(
            jid, status, **kw
        ),
        prepare_config=_prepare_image_compress_config,
        enrich_stage_results=_enrich_image_compress_stage_results,
    )
)
