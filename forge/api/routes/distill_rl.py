"""Distill-RL pipeline job routes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from forge.api.deps import get_distill_rl_orchestrator
from forge.api.http_errors import raise_forbidden
from forge.api.routes._jobs import (
    enrich_stage_results,
    resolve_linked_training_job,
    validate_pipeline_paths,
)
from forge.api.routes._pipeline import (
    StagePipelineRouterConfig,
    build_stage_pipeline_router,
)
from forge.config import ForgeSettings
from forge.db.store import Database
from forge.services.user_paths import assert_user_path, is_local_filesystem_path
from seiso.distill_rl.config import PRESETS, STAGE_ORDER, merge_distill_rl_payload
from seiso.security import SecurityError

STAGE_HELP = {
    "distill": "Teacher logits → student KL distillation",
    "rollout": "Teacher chosen / student rejected preference pairs",
    "dpo": "Direct preference optimization on distilled student",
    "evaluate": "Perplexity + validation preference accuracy",
}

_DISTILL_PATH_KEYS = ("distilled_path", "prompt_library")


class DistillRLStartRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    preset: str = Field(default="smoke", description="smoke | reproducible | full")
    config_file: str | None = None
    stages: list[str] | None = None
    teacher_model: str | None = None
    student_model: str | None = None
    distilled_path: str | None = None
    distill_steps: int | None = None
    rollout_max_prompts: int | None = None
    dpo_epochs: int | None = None
    prompt_library: str | None = None
    require_thinking_trace: bool = True
    thinking_instruction: str | None = None
    verifiable_outcome_rewards: bool = Field(
        default=True,
        description=(
            "Outcome RL for grounded sources (required true). "
            "Ignored for preference_source=teacher_style (always false)."
        ),
    )
    grpo_group_size: int = Field(default=4, ge=2)
    benchmark_verifiable: bool = True
    benchmark_tasks: list[str] = Field(
        default_factory=lambda: ["gsm8k", "gpqa", "aime"]
    )
    seeds: list[int] | None = None
    seed: int = 42
    deterministic: bool = True
    evaluate_teacher: bool = False
    hash_run_id: bool = False
    link_training_job_id: str | None = None
    auto_sweep: bool = Field(
        default=True,
        description="Grid-search DPO hyperparameters before the full alignment run.",
    )
    sweep_config: str | None = None


def _assert_local_dataset_ref(data_dir, user_id: str, config: dict[str, Any]) -> None:
    """Sandbox local ``dataset_ref`` refs (Hub ids are left alone)."""
    ref = config.get("dataset_ref") or config.get("hf_dataset")
    if not ref or not is_local_filesystem_path(ref):
        return
    try:
        assert_user_path(data_dir, user_id, ref)
    except SecurityError as exc:
        raise_forbidden(exc)


async def _prepare_distill_rl_config(
    body: BaseModel,
    db: Database,
    user_id: str,
    settings: ForgeSettings,
) -> dict[str, Any]:
    req = body
    assert isinstance(req, DistillRLStartRequest)
    config = req.model_dump(exclude_none=True)

    if req.link_training_job_id:
        await resolve_linked_training_job(
            db,
            user_id,
            req.link_training_job_id,
            config,
            path_key="distilled_path",
        )

    # Validate config_file location + body paths first.
    validate_pipeline_paths(
        settings.data_dir,
        user_id,
        config,
        config_file=req.config_file,
        path_keys=_DISTILL_PATH_KEYS,
    )
    # Merge file contents before further path checks so config_file cannot
    # smuggle a cross-tenant dataset_ref / prompt_library past the gate.
    config = merge_distill_rl_payload(config)
    validate_pipeline_paths(
        settings.data_dir,
        user_id,
        config,
        path_keys=_DISTILL_PATH_KEYS,
    )
    _assert_local_dataset_ref(settings.data_dir, user_id, config)

    return config


router = build_stage_pipeline_router(
    StagePipelineRouterConfig(
        prefix="/distill-rl",
        tags=("distill-rl",),
        audit_event_name="distill_rl_start",
        export_registry_key="distill_rl",
        presets=PRESETS,
        stage_order=STAGE_ORDER,
        stage_help=STAGE_HELP,
        model_defaults={
            "teacher_model": PRESETS["smoke"]["teacher_model"],
            "student_model": PRESETS["smoke"]["student_model"],
        },
        start_request_model=DistillRLStartRequest,
        get_orchestrator=get_distill_rl_orchestrator,
        list_jobs=lambda db, uid: db.list_distill_rl_jobs(uid),
        get_job=lambda db, jid, uid: db.get_distill_rl_job(jid, uid),
        create_job=lambda db, uid, cfg, jid: db.create_distill_rl_job(
            uid, cfg, job_id=jid
        ),
        update_status=lambda db, jid, status, **kw: db.update_distill_rl_job_status(
            jid, status, **kw
        ),
        prepare_config=_prepare_distill_rl_config,
        enrich_stage_results=lambda result: enrich_stage_results(
            result, "paper_bundle"
        ),
    )
)
