"""Configuration for distill → rollout → DPO pipelines."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from seiso.bundled.config_builder import (
    job_output_root,
    resolve_preset,
    validate_stages,
)

STAGE_ORDER = ("distill", "rollout", "dpo", "evaluate")

# Preference construction modes for Distill-RL rollout → DPO.
# Product sources only (no code_corpus / synthetic_code training path).
PREFERENCE_SOURCES = (
    "dataset",
    "data_designer",
    "grounded_library",
    "teacher_style",
)

# Minimum grounded corpus sizes (CI smoke is the only intentionally tiny path).
DATA_GEN_FLOOR_SMOKE = 32
DATA_GEN_FLOOR_REPRODUCIBLE = 256
DATA_GEN_FLOOR_FULL = 2048
GROUNDED_LIBRARY_FLOOR = 256

# Back-compat aliases used by older configs/tests.
CORPUS_FLOOR_SMOKE = DATA_GEN_FLOOR_SMOKE
CORPUS_FLOOR_REPRODUCIBLE = DATA_GEN_FLOOR_REPRODUCIBLE
CORPUS_FLOOR_FULL = DATA_GEN_FLOOR_FULL

# CI fixture (not a training corpus).
_SMOKE_FIXTURE_REL = Path("data") / "distill_verifiable_prompts.jsonl"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def smoke_fixture_prompt_library() -> Path:
    return _repo_root() / _SMOKE_FIXTURE_REL


def allow_tiny_rl(*, preset: str | None = None) -> bool:
    if os.environ.get("SEISO_ALLOW_TINY_RL", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return True
    return str(preset or "").strip().lower() == "smoke"


PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        # CI / agent fixture only — not a meaningful Distill-RL training preset.
        "stages": ["distill", "rollout", "dpo", "evaluate"],
        "teacher_model": "openai-community/gpt2",
        "student_model": "openai-community/gpt2",
        "distill_steps": 2,
        "max_train_samples": 8,
        "preference_source": "grounded_library",
        "prompt_library": str(_SMOKE_FIXTURE_REL),
        "data_gen_count": DATA_GEN_FLOOR_SMOKE,
        "rollout_max_prompts": DATA_GEN_FLOOR_SMOKE,
        "rollout_max_new_tokens": 32,
        "dpo_epochs": 1,
        "dpo_max_steps": 2,
        "dpo_save_steps": 1000,
        "dpo_learning_rate": 5e-6,
        "dpo_gradient_accumulation_steps": 4,
        "train_val_fraction": 0.75,
        "eval_max_prompts": 4,
        "align_distill_with_prompts": True,
        "verifiable_outcome_rewards": True,
    },
    "reproducible": {
        "stages": ["distill", "rollout", "dpo", "evaluate"],
        "teacher_model": "openai-community/gpt2",
        "student_model": "openai-community/gpt2",
        "teacher_revision": "main",
        "student_revision": "main",
        "distill_steps": 50,
        "max_train_samples": 32,
        "preference_source": "dataset",
        "data_gen_count": DATA_GEN_FLOOR_REPRODUCIBLE,
        "rollout_max_prompts": DATA_GEN_FLOOR_REPRODUCIBLE,
        "rollout_max_new_tokens": 128,
        "dpo_epochs": 1,
        "dpo_max_steps": 20,
        "dpo_save_steps": 50,
        "dpo_learning_rate": 5e-6,
        "dpo_gradient_accumulation_steps": 8,
        "train_val_fraction": 0.85,
        "eval_max_prompts": 32,
        "align_distill_with_prompts": True,
        "verifiable_outcome_rewards": True,
        "seeds": [13, 42, 99],
    },
    "full": {
        "stages": ["distill", "rollout", "dpo", "evaluate"],
        "teacher_model": "codellama/CodeLlama-13b-hf",
        "student_model": "codellama/CodeLlama-7b-hf",
        "distill_steps": 500,
        "max_train_samples": None,
        "preference_source": "dataset",
        "data_gen_count": DATA_GEN_FLOOR_FULL,
        "rollout_max_prompts": DATA_GEN_FLOOR_FULL,
        "rollout_max_new_tokens": 256,
        "dpo_epochs": 1,
        "dpo_max_steps": None,
        "dpo_save_steps": 200,
        "dpo_learning_rate": 2e-6,
        "dpo_gradient_accumulation_steps": 8,
        "train_val_fraction": 0.85,
        "eval_max_prompts": 64,
        "align_distill_with_prompts": True,
        "verifiable_outcome_rewards": True,
        "use_chat_template": True,
    },
}


class DistillRLConfig(BaseModel):
    job_id: str
    user_id: str
    output_root: Path
    preset: str = "smoke"
    stages: list[str] = Field(
        default_factory=lambda: ["distill", "rollout", "dpo", "evaluate"]
    )
    teacher_model: str
    student_model: str
    teacher_revision: str | None = None
    student_revision: str | None = None
    distilled_path: Path | None = None
    # User data root for local dataset_ref / prompt_library sandbox checks.
    sandbox_root: Path | None = None
    seed: int = 42
    deterministic: bool = True
    hash_run_id: bool = False

    distill_steps: int = 2
    max_train_samples: int | None = 8
    distill_alpha: float = 0.5
    distill_temperature: float = 2.0
    align_distill_with_prompts: bool = True
    require_thinking_trace: bool = True
    thinking_instruction: str = (
        "Show your reasoning in <think>...</think>, then give the final answer."
    )

    # dataset (research default) | data_designer (opt-in) | grounded_library | teacher_style
    preference_source: Literal[
        "dataset",
        "data_designer",
        "grounded_library",
        "teacher_style",
    ] = "dataset"
    # Prompt count for synth / HF materialization.
    data_gen_count: int = DATA_GEN_FLOOR_SMOKE
    # Deprecated alias of data_gen_count (accepted in payloads).
    corpus_count: int | None = None
    prompt_library_path: Path | None = None
    # HF hub id or local path when preference_source=dataset.
    dataset_ref: str | None = Field(
        default=None,
        validation_alias=AliasChoices("dataset_ref", "hf_dataset"),
    )
    dataset_split: str = "train"
    dataset_revision: str = "main"
    prompt_field: str | None = None
    answer_field: str | None = None
    tests_field: str | None = None
    preprocess_dataset: bool = True
    deduplicate_dataset: bool = True
    data_gen_mix: str = "numeric:0.7,choice:0.3"
    data_gen_difficulty: str = "easy:0.35,medium:0.45,hard:0.20"
    rollout_max_prompts: int = DATA_GEN_FLOOR_SMOKE
    rollout_max_new_tokens: int = 32
    rollout_temperature: float = 0.7
    # Derived from preference_source in the validator: grounded sources require
    # True; teacher_style is always False. Request overrides that disagree raise.
    verifiable_outcome_rewards: bool = True
    grpo_group_size: int = 4
    use_chat_template: bool | None = None
    trust_remote_code: bool = False
    train_val_fraction: float = 0.85
    # Optional OpenAI-compatible endpoint for Data Designer LLM streams.
    data_designer_base_url: str | None = None
    data_designer_model: str | None = None

    dpo_beta: float = 0.1
    dpo_epochs: int = 1
    dpo_learning_rate: float = 5e-6
    dpo_batch_size: int = 1
    dpo_gradient_accumulation_steps: int = 8
    dpo_max_steps: int | None = None
    dpo_save_steps: int = 200
    dpo_use_lora: bool = True
    dpo_use_qlora: bool = False
    # False = sum token log-probs (Rafailov / DPOSettings default); True = mean (length-normalized).
    dpo_average_log_prob: bool = False
    dpo_warmup_ratio: float = 0.1
    dpo_weight_decay: float = 0.01
    dpo_max_grad_norm: float = 0.3
    dpo_output_dir_override: Path | None = None

    eval_max_prompts: int = 8
    evaluate_teacher: bool = False
    benchmark_verifiable: bool = True
    benchmark_tasks: list[str] = Field(
        default_factory=lambda: ["gsm8k", "gpqa", "aime"]
    )

    @field_validator("preference_source", mode="before")
    @classmethod
    def _normalize_preference_source(cls, value: Any) -> Any:
        if value is None:
            return value
        text = str(value).strip().lower()
        if text == "hf_dataset":
            return "dataset"
        return text

    @model_validator(mode="after")
    def _validate_grounded_floors(self) -> DistillRLConfig:
        source = str(self.preference_source).strip().lower()
        if source in {"code_corpus", "synthetic_code"}:
            raise ValueError(
                f"preference_source={source!r} is not a product training path. "
                "Use dataset, data_designer, grounded_library, or teacher_style. "
                "CI: preference_source=grounded_library + fixture + SEISO_ALLOW_TINY_RL=1 "
                "(preset=smoke wires the fixture automatically)."
            )
        if source not in PREFERENCE_SOURCES:
            raise ValueError(
                f"preference_source must be one of {PREFERENCE_SOURCES}; got {source!r}"
            )
        object.__setattr__(self, "preference_source", source)

        count = int(self.data_gen_count)
        if self.corpus_count is not None:
            count = int(self.corpus_count)
        object.__setattr__(self, "data_gen_count", count)
        object.__setattr__(self, "corpus_count", count)

        tiny = allow_tiny_rl(preset=self.preset)
        floor = _data_gen_floor_for_preset(self.preset)
        if source in {"data_designer", "dataset"}:
            if count < floor and not tiny:
                raise ValueError(
                    f"data_gen_count={count} below floor {floor} for "
                    f"preset={self.preset!r}. Use data_gen_count>={floor}, "
                    "preset=smoke for CI, or SEISO_ALLOW_TINY_RL=1 for agent experiments."
                )
            if count < 1:
                raise ValueError("data_gen_count must be positive")
        if source == "grounded_library" and self.prompt_library_path is None:
            raise ValueError(
                "preference_source=grounded_library requires prompt_library "
                "(JSON/JSONL with answer and/or tests on each prompt)"
            )
        if source == "dataset" and not (self.dataset_ref or self.prompt_library_path):
            raise ValueError(
                "preference_source=dataset requires dataset_ref or "
                "prompt_library (HF hub id or local path with answer/tests)"
            )
        # Single source of truth: outcome mode follows preference_source.
        if source == "teacher_style":
            # Open-style DPO; default request flag is True — normalize off.
            object.__setattr__(self, "verifiable_outcome_rewards", False)
        elif (
            source in {"dataset", "data_designer", "grounded_library"}
            and not self.verifiable_outcome_rewards
        ):
            raise ValueError(
                f"preference_source={source!r} requires "
                "verifiable_outcome_rewards=true (outcome RL). "
                "Use preference_source=teacher_style for teacher≻student style DPO."
            )

        # Held-out preference split required for product runs.
        frac = float(self.train_val_fraction)
        if not tiny:
            if frac <= 0.0 or frac >= 1.0:
                raise ValueError(
                    "train_val_fraction must be in (0, 1) for product Distill-RL "
                    "so preferences_val.jsonl is a held-out split. "
                    "CI/smoke may use SEISO_ALLOW_TINY_RL=1."
                )
            if "rollout" in self.stages and "evaluate" not in self.stages:
                raise ValueError(
                    "product Distill-RL runs that build preferences must include "
                    "the evaluate stage (held-out val preference accuracy). "
                    "CI/smoke may set SEISO_ALLOW_TINY_RL=1."
                )
        elif frac <= 0.0 or frac > 1.0:
            raise ValueError("train_val_fraction must be in (0, 1]")
        return self

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


def _data_gen_floor_for_preset(preset: str) -> int:
    key = str(preset or "smoke").strip().lower()
    if key == "smoke":
        return DATA_GEN_FLOOR_SMOKE
    if key == "full":
        return DATA_GEN_FLOOR_FULL
    return DATA_GEN_FLOOR_REPRODUCIBLE


# Back-compat name.
_corpus_floor_for_preset = _data_gen_floor_for_preset


def resolve_preference_source(
    merged: dict[str, Any],
    preset: dict[str, Any],
    *,
    prompt_path: Path | None,
) -> str:
    """Resolve preference_source with safe defaults for meaningful Distill-RL."""
    raw = merged.get("preference_source", preset.get("preference_source"))
    if raw is not None and str(raw).strip():
        source = str(raw).strip().lower()
        if source == "hf_dataset":
            source = "dataset"
    elif merged.get("dataset_ref") or merged.get("hf_dataset") or (
        prompt_path is None
        and merged.get("prompt_library")
        and not Path(str(merged.get("prompt_library"))).expanduser().is_file()
    ):
        source = "dataset"
    elif prompt_path is not None and not bool(
        merged.get(
            "verifiable_outcome_rewards",
            preset.get("verifiable_outcome_rewards", True),
        )
    ):
        source = "teacher_style"
    elif prompt_path is not None:
        source = "grounded_library"
    else:
        source = str(preset.get("preference_source") or "dataset").strip().lower()
        if source == "hf_dataset":
            source = "dataset"
    if source in {"code_corpus", "synthetic_code"}:
        raise ValueError(
            f"preference_source={source!r} is not a product training path. "
            "Use dataset, data_designer, grounded_library, or teacher_style."
        )
    if source not in PREFERENCE_SOURCES:
        raise ValueError(
            f"preference_source must be one of {PREFERENCE_SOURCES}; got {source!r}"
        )
    return source


def merge_distill_rl_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Merge optional ``config_file`` under request overrides (body wins)."""
    merged = dict(payload)
    if config_path := _resolve_config_file(payload.get("config_file")):
        file_payload = _load_config_file(config_path)
        merged = {**file_payload, **merged}
    return merged


# Compat alias used by older call sites / tests.
_merged_payload = merge_distill_rl_payload


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
    return any(
        marker in lowered for marker in ("instruct", "chat", "-it", "_it", "gemma")
    )


def build_distill_rl_config(
    *,
    job_id: str,
    user_id: str,
    data_dir: Path,
    payload: dict[str, Any],
) -> DistillRLConfig:
    merged = _merged_payload(payload)
    preset_name, preset = resolve_preset(PRESETS, str(merged.get("preset", "smoke")))
    stages = list(
        merged.get("stages") or preset.get("stages") or PRESETS["smoke"]["stages"]
    )
    validate_stage_sequence(stages)

    hash_run_id = bool(merged.get("hash_run_id", False))
    output_root = job_output_root(
        data_dir,
        "distill_rl",
        user_id,
        _job_storage_id(job_id, hash_run_id=hash_run_id),
    )
    seed = int(merged.get("seed", 42))

    prompt_library = merged.get("prompt_library", preset.get("prompt_library"))
    prompt_path = Path(prompt_library).expanduser() if prompt_library else None
    if prompt_path is not None and not prompt_path.is_file():
        # Allow repo-relative fixtures (e.g. data/distill_verifiable_prompts.jsonl).
        alt = _repo_root() / prompt_path
        if alt.is_file():
            prompt_path = alt
    # Smoke wires the CI fixture when grounded_library has no path yet.
    if (
        preset_name == "smoke"
        and prompt_path is None
        and str(
            merged.get("preference_source", preset.get("preference_source"))
        ).strip().lower()
        in {"", "grounded_library"}
    ):
        prompt_path = smoke_fixture_prompt_library()

    preference_source = resolve_preference_source(
        merged, preset, prompt_path=prompt_path
    )
    floor = _data_gen_floor_for_preset(preset_name)
    data_gen_count = int(
        merged.get(
            "data_gen_count",
            merged.get(
                "corpus_count",
                preset.get(
                    "data_gen_count",
                    preset.get(
                        "corpus_count",
                        merged.get(
                            "rollout_max_prompts",
                            preset.get("rollout_max_prompts", floor),
                        ),
                    ),
                ),
            ),
        )
    )

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
        sandbox_root=Path(data_dir).expanduser().resolve(),
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
        max_train_samples=merged.get(
            "max_train_samples", preset.get("max_train_samples")
        ),
        distill_alpha=float(merged.get("distill_alpha", 0.5)),
        distill_temperature=float(merged.get("distill_temperature", 2.0)),
        align_distill_with_prompts=bool(
            merged.get(
                "align_distill_with_prompts",
                preset.get("align_distill_with_prompts", True),
            )
        ),
        require_thinking_trace=bool(merged.get("require_thinking_trace", True)),
        thinking_instruction=str(
            merged.get(
                "thinking_instruction",
                "Show your reasoning in <think>...</think>, then give the final answer.",
            )
        ),
        preference_source=preference_source,
        data_gen_count=data_gen_count,
        prompt_library_path=prompt_path,
        dataset_ref=(
            str(merged.get("dataset_ref") or merged.get("hf_dataset") or "").strip()
            or None
        ),
        dataset_split=str(merged.get("dataset_split", "train")),
        dataset_revision=str(merged.get("dataset_revision", "main")),
        prompt_field=(
            str(merged["prompt_field"]) if merged.get("prompt_field") else None
        ),
        answer_field=(
            str(merged["answer_field"]) if merged.get("answer_field") else None
        ),
        tests_field=(str(merged["tests_field"]) if merged.get("tests_field") else None),
        preprocess_dataset=bool(merged.get("preprocess_dataset", True)),
        deduplicate_dataset=bool(merged.get("deduplicate_dataset", True)),
        data_gen_mix=str(merged.get("data_gen_mix", "numeric:0.7,choice:0.3")),
        data_gen_difficulty=str(
            merged.get("data_gen_difficulty", "easy:0.35,medium:0.45,hard:0.20")
        ),
        rollout_max_prompts=int(
            merged.get(
                "rollout_max_prompts",
                preset.get("rollout_max_prompts", data_gen_count),
            )
        ),
        rollout_max_new_tokens=int(
            merged.get(
                "rollout_max_new_tokens", preset.get("rollout_max_new_tokens", 32)
            )
        ),
        rollout_temperature=float(merged.get("rollout_temperature", 0.7)),
        verifiable_outcome_rewards=bool(
            merged.get(
                "verifiable_outcome_rewards",
                preset.get("verifiable_outcome_rewards", True),
            )
        ),
        grpo_group_size=int(merged.get("grpo_group_size", 4)),
        use_chat_template=use_chat_template,
        trust_remote_code=bool(merged.get("trust_remote_code", False)),
        train_val_fraction=float(
            merged.get("train_val_fraction", preset.get("train_val_fraction", 0.85))
        ),
        data_designer_base_url=(
            str(merged["data_designer_base_url"]).strip()
            if merged.get("data_designer_base_url")
            else None
        ),
        data_designer_model=(
            str(merged["data_designer_model"]).strip()
            if merged.get("data_designer_model")
            else None
        ),
        dpo_beta=float(merged.get("dpo_beta", preset.get("dpo_beta", 0.1))),
        dpo_epochs=int(merged.get("dpo_epochs", preset.get("dpo_epochs", 1))),
        dpo_learning_rate=float(
            merged.get("dpo_learning_rate", preset.get("dpo_learning_rate", 5e-6))
        ),
        dpo_batch_size=int(
            merged.get("dpo_batch_size", preset.get("dpo_batch_size", 1))
        ),
        dpo_gradient_accumulation_steps=int(
            merged.get(
                "dpo_gradient_accumulation_steps",
                preset.get("dpo_gradient_accumulation_steps", 8),
            )
        ),
        dpo_max_steps=merged.get("dpo_max_steps", preset.get("dpo_max_steps")),
        dpo_save_steps=int(
            merged.get("dpo_save_steps", preset.get("dpo_save_steps", 200))
        ),
        dpo_use_lora=bool(merged.get("dpo_use_lora", True)),
        dpo_use_qlora=bool(merged.get("dpo_use_qlora", False)),
        dpo_average_log_prob=bool(
            merged.get("dpo_average_log_prob", preset.get("dpo_average_log_prob", False))
        ),
        dpo_warmup_ratio=float(
            merged.get("dpo_warmup_ratio", preset.get("dpo_warmup_ratio", 0.1))
        ),
        dpo_weight_decay=float(
            merged.get("dpo_weight_decay", preset.get("dpo_weight_decay", 0.01))
        ),
        dpo_max_grad_norm=float(
            merged.get("dpo_max_grad_norm", preset.get("dpo_max_grad_norm", 0.3))
        ),
        eval_max_prompts=int(
            merged.get("eval_max_prompts", preset.get("eval_max_prompts", 8))
        ),
        evaluate_teacher=bool(merged.get("evaluate_teacher", False)),
        benchmark_verifiable=bool(merged.get("benchmark_verifiable", True)),
        benchmark_tasks=[
            str(task).lower()
            for task in merged.get("benchmark_tasks", ["gsm8k", "gpqa", "aime"])
        ],
    )
