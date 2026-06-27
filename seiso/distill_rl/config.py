"""Configuration for distill → rollout → DPO pipelines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, Field

from seiso.vendor.config_builder import job_output_root, resolve_preset, validate_stages

STAGE_ORDER = ("distill", "rollout", "dpo", "evaluate")

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "stages": ["distill", "rollout", "dpo", "evaluate"],
        "teacher_model": "openai-community/gpt2",
        "student_model": "openai-community/gpt2",
        "distill_steps": 2,
        "max_train_samples": 8,
        "rollout_max_prompts": 4,
        "rollout_max_new_tokens": 32,
        "dpo_epochs": 1,
        "dpo_max_steps": 2,
        "dpo_save_steps": 1000,
        "dpo_learning_rate": 5e-6,
        "dpo_gradient_accumulation_steps": 4,
        "train_val_fraction": 0.75,
        "eval_max_prompts": 4,
        "align_distill_with_prompts": True,
    },
    "reproducible": {
        "stages": ["distill", "rollout", "dpo", "evaluate"],
        "teacher_model": "openai-community/gpt2",
        "student_model": "openai-community/gpt2",
        "teacher_revision": "main",
        "student_revision": "main",
        "distill_steps": 50,
        "max_train_samples": 32,
        "rollout_max_prompts": 32,
        "rollout_max_new_tokens": 128,
        "dpo_epochs": 1,
        "dpo_max_steps": 20,
        "dpo_save_steps": 50,
        "dpo_learning_rate": 5e-6,
        "dpo_gradient_accumulation_steps": 8,
        "train_val_fraction": 0.85,
        "eval_max_prompts": 32,
        "align_distill_with_prompts": True,
        "seeds": [13, 42, 99],
    },
    "full": {
        "stages": ["distill", "rollout", "dpo", "evaluate"],
        "teacher_model": "codellama/CodeLlama-13b-hf",
        "student_model": "codellama/CodeLlama-7b-hf",
        "distill_steps": 500,
        "max_train_samples": None,
        "rollout_max_prompts": 64,
        "rollout_max_new_tokens": 256,
        "dpo_epochs": 1,
        "dpo_max_steps": None,
        "dpo_save_steps": 200,
        "dpo_learning_rate": 2e-6,
        "dpo_gradient_accumulation_steps": 8,
        "train_val_fraction": 0.85,
        "eval_max_prompts": 64,
        "align_distill_with_prompts": True,
        "use_chat_template": True,
    },
}


class DistillRLConfig(BaseModel):
    job_id: str
    user_id: str
    output_root: Path
    preset: str = "smoke"
    stages: list[str] = Field(default_factory=lambda: ["distill", "rollout", "dpo", "evaluate"])
    teacher_model: str
    student_model: str
    teacher_revision: str | None = None
    student_revision: str | None = None
    distilled_path: Path | None = None
    seed: int = 42
    deterministic: bool = True
    hash_run_id: bool = False

    distill_steps: int = 2
    max_train_samples: int | None = 8
    distill_alpha: float = 0.5
    distill_temperature: float = 2.0
    align_distill_with_prompts: bool = True

    prompt_library_path: Path | None = None
    rollout_max_prompts: int = 4
    rollout_max_new_tokens: int = 32
    rollout_temperature: float = 0.7
    use_chat_template: bool | None = None
    train_val_fraction: float = 0.85

    dpo_beta: float = 0.1
    dpo_epochs: int = 1
    dpo_learning_rate: float = 5e-6
    dpo_batch_size: int = 1
    dpo_gradient_accumulation_steps: int = 8
    dpo_max_steps: int | None = None
    dpo_save_steps: int = 200
    dpo_use_lora: bool = True
    dpo_use_qlora: bool = False
    dpo_average_log_prob: bool = True
    dpo_warmup_ratio: float = 0.1
    dpo_weight_decay: float = 0.01
    dpo_max_grad_norm: float = 0.3
    dpo_output_dir_override: Path | None = None

    eval_max_prompts: int = 8
    evaluate_teacher: bool = False

    @property
    def distilled_dir(self) -> Path:
        return self.output_root / "distilled"

    @property
    def preferences_dir(self) -> Path:
        return self.output_root / "preferences"

    @property
    def preferences_train_path(self) -> Path:
        return self.preferences_dir / "preferences_train.jsonl"

    @property
    def preferences_val_path(self) -> Path:
        return self.preferences_dir / "preferences_val.jsonl"

    @property
    def preferences_path(self) -> Path:
        return self.output_root / "preferences.jsonl"

    @property
    def dpo_output_dir(self) -> Path:
        if self.dpo_output_dir_override is not None:
            return self.dpo_output_dir_override
        return self.output_root / "dpo"

    @property
    def evaluation_dir(self) -> Path:
        return self.output_root / "evaluation"


def validate_stage_sequence(stages: list[str]) -> None:
    """Ensure stages are known and appear in pipeline order."""
    validate_stages(stages, STAGE_ORDER)
    indices = [STAGE_ORDER.index(stage) for stage in stages]
    if indices != sorted(indices):
        raise ValueError(f"Stages must follow order {STAGE_ORDER}; got {stages}")


def _merged_payload(payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(payload)
    if config_path := _resolve_config_file(payload.get("config_file")):
        file_payload = _load_config_file(config_path)
        merged = {**file_payload, **merged}
    return merged


def resolve_job_seeds(payload: dict[str, Any]) -> list[int] | None:
    """Resolve multi-seed list from payload, config file, or preset."""
    raw = payload.get("seeds")
    if isinstance(raw, list) and len(raw) > 1:
        return [int(s) for s in raw]

    merged = _merged_payload(payload)
    preset_name = str(merged.get("preset", "smoke"))
    _, preset = resolve_preset(PRESETS, preset_name)
    preset_seeds = merged.get("seeds") or preset.get("seeds")
    if isinstance(preset_seeds, list) and len(preset_seeds) > 1:
        return [int(s) for s in preset_seeds]
    return None


def _job_storage_id(job_id: str, *, hash_run_id: bool) -> str:
    if hash_run_id:
        return hashlib.sha256(job_id.encode()).hexdigest()[:16]
    return job_id


def _load_config_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        with path.open(encoding="utf-8") as handle:
            return cast(dict[str, Any], yaml.safe_load(handle) or {})
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _resolve_config_file(config_file: str | None) -> Path | None:
    if not config_file:
        return None
    path = Path(config_file).expanduser()
    if path.is_file():
        return path
    repo_path = Path(__file__).resolve().parents[2] / "configs" / config_file
    return repo_path if repo_path.is_file() else None


def infer_use_chat_template(model_id: str, explicit: bool | None) -> bool:
    if explicit is not None:
        return explicit
    lowered = model_id.lower()
    return any(marker in lowered for marker in ("instruct", "chat", "-it", "_it", "gemma"))


def build_distill_rl_config(
    *,
    job_id: str,
    user_id: str,
    data_dir: Path,
    payload: dict[str, Any],
) -> DistillRLConfig:
    merged = _merged_payload(payload)
    preset_name, preset = resolve_preset(PRESETS, str(merged.get("preset", "smoke")))
    stages = list(merged.get("stages") or preset.get("stages") or PRESETS["smoke"]["stages"])
    validate_stage_sequence(stages)

    hash_run_id = bool(merged.get("hash_run_id", False))
    output_root = job_output_root(
        data_dir,
        "distill_rl",
        user_id,
        _job_storage_id(job_id, hash_run_id=hash_run_id),
    )
    seed = int(merged.get("seed", 42))

    prompt_library = merged.get("prompt_library")
    prompt_path = Path(prompt_library).expanduser() if prompt_library else None

    distilled_raw = merged.get("distilled_path")
    distilled_path = Path(str(distilled_raw)).expanduser() if distilled_raw else None

    teacher_model = str(merged.get("teacher_model") or preset["teacher_model"])
    student_model = str(merged.get("student_model") or preset["student_model"])
    use_chat_template = infer_use_chat_template(
        student_model,
        merged.get("use_chat_template", preset.get("use_chat_template")),
    )

    return DistillRLConfig(
        job_id=job_id,
        user_id=user_id,
        output_root=output_root,
        preset=preset_name,
        stages=stages,
        teacher_model=teacher_model,
        student_model=student_model,
        teacher_revision=merged.get("teacher_revision", preset.get("teacher_revision")),
        student_revision=merged.get("student_revision", preset.get("student_revision")),
        distilled_path=distilled_path,
        seed=seed,
        deterministic=bool(merged.get("deterministic", True)),
        hash_run_id=bool(merged.get("hash_run_id", False)),
        distill_steps=int(merged.get("distill_steps", preset.get("distill_steps", 2))),
        max_train_samples=merged.get("max_train_samples", preset.get("max_train_samples")),
        distill_alpha=float(merged.get("distill_alpha", 0.5)),
        distill_temperature=float(merged.get("distill_temperature", 2.0)),
        align_distill_with_prompts=bool(
            merged.get("align_distill_with_prompts", preset.get("align_distill_with_prompts", True))
        ),
        prompt_library_path=prompt_path,
        rollout_max_prompts=int(
            merged.get("rollout_max_prompts", preset.get("rollout_max_prompts", 4))
        ),
        rollout_max_new_tokens=int(
            merged.get("rollout_max_new_tokens", preset.get("rollout_max_new_tokens", 32))
        ),
        rollout_temperature=float(merged.get("rollout_temperature", 0.7)),
        use_chat_template=use_chat_template,
        train_val_fraction=float(
            merged.get("train_val_fraction", preset.get("train_val_fraction", 0.85))
        ),
        dpo_beta=float(merged.get("dpo_beta", preset.get("dpo_beta", 0.1))),
        dpo_epochs=int(merged.get("dpo_epochs", preset.get("dpo_epochs", 1))),
        dpo_learning_rate=float(
            merged.get("dpo_learning_rate", preset.get("dpo_learning_rate", 5e-6))
        ),
        dpo_batch_size=int(merged.get("dpo_batch_size", preset.get("dpo_batch_size", 1))),
        dpo_gradient_accumulation_steps=int(
            merged.get("dpo_gradient_accumulation_steps", preset.get("dpo_gradient_accumulation_steps", 8))
        ),
        dpo_max_steps=merged.get("dpo_max_steps", preset.get("dpo_max_steps")),
        dpo_save_steps=int(merged.get("dpo_save_steps", preset.get("dpo_save_steps", 200))),
        dpo_use_lora=bool(merged.get("dpo_use_lora", True)),
        dpo_use_qlora=bool(merged.get("dpo_use_qlora", False)),
        dpo_average_log_prob=bool(
            merged.get("dpo_average_log_prob", preset.get("dpo_average_log_prob", True))
        ),
        dpo_warmup_ratio=float(merged.get("dpo_warmup_ratio", preset.get("dpo_warmup_ratio", 0.1))),
        dpo_weight_decay=float(merged.get("dpo_weight_decay", preset.get("dpo_weight_decay", 0.01))),
        dpo_max_grad_norm=float(merged.get("dpo_max_grad_norm", preset.get("dpo_max_grad_norm", 0.3))),
        eval_max_prompts=int(merged.get("eval_max_prompts", preset.get("eval_max_prompts", 8))),
        evaluate_teacher=bool(merged.get("evaluate_teacher", False)),
    )
