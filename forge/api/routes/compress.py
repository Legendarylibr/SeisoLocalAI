"""Code Llama compression pipeline job routes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from forge.api.deps import get_compress_orchestrator
from forge.api.routes._jobs import (
    enrich_stage_results,
    resolve_linked_training_job,
    validate_pipeline_paths,
)
from forge.api.routes._pipeline import StagePipelineRouterConfig, build_stage_pipeline_router
from forge.config import ForgeSettings
from forge.db.store import Database
from seiso.compress.config_builder import PRESETS, STAGE_ORDER, get_compress_model_defaults

STAGE_HELP = {
    "distill": "Teacher → student KL distillation",
    "prune": "Shape-preserving MLP neuron masking",
    "finetune": "Post-prune recovery fine-tuning",
    "evaluate": "Perplexity + speed smoke check",
    "export": "vLLM/Docker/GGUF helper scripts",
    "quantize_gptq": "GPTQ 4-bit (requires pip install seiso[compress-quant])",
    "quantize_awq": "AWQ 4-bit (requires pip install seiso[compress-quant])",
}


class CompressStartRequest(BaseModel):
    preset: str = Field(
        default="smoke",
        description="smoke | full | distill_only | prune_recover | quantize",
    )
    stages: list[str] | None = None
    config_file: str | None = None
    teacher_model: str | None = None
    student_model: str | None = None
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


async def _prepare_compress_config(
    body: BaseModel,
    db: Database,
    user_id: str,
    settings: ForgeSettings,
) -> dict[str, Any]:
    req = body
    assert isinstance(req, CompressStartRequest)
    config = req.model_dump()
    defaults = get_compress_model_defaults()
    if not config.get("teacher_model"):
        config["teacher_model"] = defaults["teacher_model"]
    if not config.get("student_model"):
        config["student_model"] = defaults["student_model"]

    if req.link_training_job_id:
        await resolve_linked_training_job(
            db,
            user_id,
            req.link_training_job_id,
            config,
            path_key="model_dir",
            preset_when="smoke",
            preset_override="prune_recover",
        )

    validate_pipeline_paths(
        settings.data_dir,
        user_id,
        config,
        config_file=req.config_file,
        path_keys=("model_dir",),
    )

    return config


router = build_stage_pipeline_router(
    StagePipelineRouterConfig(
        prefix="/compress",
        tags=("compress",),
        audit_event_name="compress_start",
        export_registry_key="compressed",
        presets=PRESETS,
        stage_order=STAGE_ORDER,
        stage_help=STAGE_HELP,
        model_defaults=get_compress_model_defaults(),
        start_request_model=CompressStartRequest,
        get_orchestrator=get_compress_orchestrator,
        list_jobs=lambda db, uid: db.list_compress_jobs(uid),
        get_job=lambda db, jid, uid: db.get_compress_job(jid, uid),
        create_job=lambda db, uid, cfg, jid: db.create_compress_job(uid, cfg, job_id=jid),
        update_status=lambda db, jid, status, **kw: db.update_compress_job_status(
            jid, status, **kw
        ),
        prepare_config=_prepare_compress_config,
        enrich_stage_results=enrich_stage_results,
    )
)
