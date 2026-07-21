"""Build pipeline config for Seiso compression jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from seiso.bundled.config_builder import (
    job_output_root,
    resolve_config_file_path,
    resolve_preset,
    validate_stages,
)
from seiso.compress.bootstrap import bundle_root, ensure_codellama_compress_importable

STAGE_ORDER = (
    "distill",
    "prune",
    "finetune",
    "evaluate",
    "export",
    "quantize_gptq",
    "quantize_awq",
)

_MODEL_DEFAULTS: dict[str, str] | None = None


def get_compress_model_defaults() -> dict[str, str]:
    """Bundled DistillConfig defaults — single source for API and pipeline builder."""
    global _MODEL_DEFAULTS
    if _MODEL_DEFAULTS is None:
        ensure_codellama_compress_importable()
        from seiso.codellama_compress.config import DistillConfig

        dc = DistillConfig()
        _MODEL_DEFAULTS = {
            "teacher_model": dc.teacher_model,
            "student_model": dc.student_model,
        }
    return dict(_MODEL_DEFAULTS)


PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "stages": ["distill", "prune", "finetune", "evaluate", "export"],
        "distill_steps": 2,
        "finetune_steps": 2,
        "prune_ratio": 0.1,
        "max_train_samples": 32,
        "calibration_samples": 8,
    },
    "full": {
        "stages": ["distill", "prune", "finetune", "evaluate", "export"],
        "distill_steps": 1000,
        "finetune_steps": 500,
        "prune_ratio": 0.25,
    },
    "distill_only": {
        "stages": ["distill", "evaluate"],
        "distill_steps": 500,
    },
    "prune_recover": {
        "stages": ["prune", "finetune", "evaluate", "export"],
        "finetune_steps": 500,
        "prune_ratio": 0.25,
    },
    "quantize": {
        "stages": ["quantize_gptq", "evaluate", "export"],
        "calibration_samples": 128,
    },
}


def build_pipeline_config(
    *,
    job_id: str,
    user_id: str,
    data_dir: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Return normalized pipeline settings for a Forge or CLI job."""
    ensure_codellama_compress_importable()
    from seiso.codellama_compress.config import (
        DatasetConfig,
        DeterminismConfig,
        DistillConfig,
        GPTQConfig,
        merge_dataclass,
    )

    preset_name, preset = resolve_preset(PRESETS, str(payload.get("preset", "smoke")))

    config_blob: dict[str, Any] = {}
    if path := resolve_config_file_path(
        payload.get("config_file"), bundle_root=bundle_root()
    ):
        from seiso.codellama_compress.config import load_config_file

        config_blob = load_config_file(path)
        preset.update(config_blob.get("pipeline", {}))

    stages = list(
        payload.get("stages") or preset.get("stages") or PRESETS["smoke"]["stages"]
    )
    validate_stages(stages, STAGE_ORDER)

    output_root = job_output_root(data_dir, "compress", user_id, job_id)

    seed = int(payload.get("seed", 42))
    det_overrides = dict(config_blob.get("determinism") or {})
    det_overrides.update(
        {
            "seed": seed,
            "deterministic": bool(payload.get("deterministic", True)),
            "hash_run_id": bool(payload.get("hash_run_id", True)),
        }
    )
    det = merge_dataclass(DeterminismConfig(), det_overrides)
    ds_overrides = dict(config_blob.get("dataset") or {})
    ds_overrides.setdefault("seed", seed)
    if "max_train_samples" not in ds_overrides:
        ds_overrides["max_train_samples"] = payload.get(
            "max_train_samples", preset.get("max_train_samples")
        )
    ds_cfg = merge_dataclass(DatasetConfig(), ds_overrides)
    model_defaults = get_compress_model_defaults()
    distill_file = dict(config_blob.get("distill") or {})
    distill_overrides = {
        "teacher_model": str(
            payload.get("teacher_model")
            or distill_file.get("teacher_model")
            or model_defaults["teacher_model"]
        ),
        "student_model": str(
            payload.get("student_model")
            or distill_file.get("student_model")
            or model_defaults["student_model"]
        ),
        "steps": int(
            payload.get(
                "distill_steps",
                distill_file.get("steps", preset.get("distill_steps", 2)),
            )
        ),
        "trust_remote_code": bool(payload.get("trust_remote_code", False)),
    }
    for key in ("temperature", "alpha", "lr", "seq_len", "warmup_steps", "weight_decay"):
        if key in distill_file and key not in payload:
            distill_overrides[key] = distill_file[key]
    for key in ("temperature", "alpha"):
        if key in payload:
            distill_overrides[key] = payload[key]
    distill_cfg = merge_dataclass(DistillConfig(), distill_overrides)
    finetune_file = dict(config_blob.get("finetune") or {})
    finetune_cfg = merge_dataclass(
        DistillConfig(teacher_model="", alpha=0.0, temperature=1.0),
        {
            "steps": int(
                payload.get(
                    "finetune_steps",
                    finetune_file.get("steps", preset.get("finetune_steps", 2)),
                )
            ),
            "trust_remote_code": bool(payload.get("trust_remote_code", False)),
        },
    )
    gptq_cfg = merge_dataclass(
        GPTQConfig(),
        {
            "seed": seed,
            "calibration_samples": int(
                payload.get(
                    "calibration_samples", preset.get("calibration_samples", 32)
                )
            ),
        },
    )

    model_dir = payload.get("model_dir")
    if model_dir:
        model_dir = str(Path(model_dir).expanduser().resolve())
    elif stages and stages[0] != "distill":
        raise ValueError(
            f"Preset {preset_name!r} starts with {stages[0]!r}; provide --model-dir "
            "with an existing model artifact directory."
        )

    return {
        "job_id": job_id,
        "user_id": user_id,
        "preset": preset_name,
        "output_root": str(output_root.resolve()),
        "stages": stages,
        "determinism": det,
        "dataset": ds_cfg,
        "distill": distill_cfg,
        "finetune": finetune_cfg,
        "gptq": gptq_cfg,
        "awq": gptq_cfg,
        "prune": {
            "ratio": float(payload.get("prune_ratio", preset.get("prune_ratio", 0.25))),
            "method": str(payload.get("prune_method", "magnitude")),
        },
        "export": {
            "model_name": str(payload.get("export_model_name", "seiso-compressed")),
            "port": int(payload.get("export_port", 8000)),
        },
        "model_dir": model_dir,
        "env_report": bool(
            payload.get(
                "env_report",
                preset_name not in {"smoke"}
                and bool(payload.get("deterministic", True)),
            )
        ),
        "min_free_gb": payload.get("min_free_gb"),
        "max_run_dir_gb": payload.get("max_run_dir_gb"),
    }
