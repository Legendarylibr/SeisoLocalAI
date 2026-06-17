"""Build pipeline config for Seiso image compression jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from seiso.image_compress.bootstrap import _VENDOR_ROOT, ensure_sd_compress_importable

STAGE_ORDER = (
    "baseline",
    "distill_progressive",
    "distill_clip",
    "distill_cfg",
    "evaluate_distilled",
    "prune",
    "evaluate_pruned",
    "finetune",
    "evaluate_finetuned",
    "quantize",
    "evaluate_quantized",
    "optimize",
    "export_onnx",
    "export_shard",
    "lora_test",
    "report",
)

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "stages": [
            "baseline",
            "distill_progressive",
            "prune",
            "quantize",
            "evaluate_quantized",
            "report",
        ],
        "steps": 2,
        "clip_distill_steps": 2,
        "cfg_distill_steps": 2,
        "finetune_steps": 2,
        "eval_samples": 2,
        "eval_inference_steps": 4,
        "int8_calibration_samples": 4,
    },
    "full": {
        "stages": list(STAGE_ORDER),
        "steps": 800,
        "clip_distill_steps": 400,
        "cfg_distill_steps": 400,
        "finetune_steps": 200,
    },
    "distill_only": {
        "stages": [
            "baseline",
            "distill_progressive",
            "distill_clip",
            "distill_cfg",
            "evaluate_distilled",
            "report",
        ],
        "steps": 400,
        "clip_distill_steps": 200,
        "cfg_distill_steps": 200,
    },
    "prune_recover": {
        "stages": [
            "prune",
            "evaluate_pruned",
            "finetune",
            "evaluate_finetuned",
            "report",
        ],
        "finetune_steps": 200,
        "prune_ratio": 0.3,
    },
    "quantize": {
        "stages": [
            "quantize",
            "evaluate_quantized",
            "export_shard",
            "report",
        ],
        "int8_calibration_samples": 100,
    },
}


def build_pipeline_config(
    *,
    job_id: str,
    user_id: str,
    data_dir: Path,
    payload: dict[str, Any],
) -> Any:
    """Return sd_compress.config.PipelineConfig for a Forge or CLI job."""
    ensure_sd_compress_importable()
    from sd_compress.config import PipelineConfig, ensure_captions

    preset_name = str(payload.get("preset", "smoke"))
    preset = dict(PRESETS.get(preset_name, PRESETS["smoke"]))

    output_root = data_dir / "image_compress" / user_id / job_id
    output_root.mkdir(parents=True, exist_ok=True)

    data_path = payload.get("data_path")
    if not data_path:
        data_path = str(output_root / "captions.json")

    overrides: dict[str, Any] = {
        "base_model": str(payload.get("base_model", "runwayml/stable-diffusion-v1-5")),
        "data_path": str(data_path),
        "output_dir": str(output_root),
        "distill_dir": str(output_root / "distilled"),
        "prune_dir": str(output_root / "pruned"),
        "finetune_dir": str(output_root / "finetuned"),
        "quant_dir": str(output_root / "quant"),
        "export_dir": str(output_root / "export"),
        "steps": int(payload.get("steps", preset.get("steps", 2))),
        "clip_distill_steps": int(
            payload.get("clip_distill_steps", preset.get("clip_distill_steps", 2))
        ),
        "cfg_distill_steps": int(
            payload.get("cfg_distill_steps", preset.get("cfg_distill_steps", 2))
        ),
        "finetune_steps": int(payload.get("finetune_steps", preset.get("finetune_steps", 2))),
        "prune_ratio": float(payload.get("prune_ratio", preset.get("prune_ratio", 0.3))),
        "text_encoder_prune_ratio": float(
            payload.get("text_encoder_prune_ratio", preset.get("text_encoder_prune_ratio", 0.25))
        ),
        "vae_prune_ratio": float(payload.get("vae_prune_ratio", preset.get("vae_prune_ratio", 0.2))),
        "int8_calibration_samples": int(
            payload.get(
                "int8_calibration_samples",
                preset.get("int8_calibration_samples", 8),
            )
        ),
        "eval_samples": int(payload.get("eval_samples", preset.get("eval_samples", 4))),
        "eval_inference_steps": int(
            payload.get("eval_inference_steps", preset.get("eval_inference_steps", 6))
        ),
    }

    if model_dir := payload.get("model_dir"):
        overrides["distill_dir"] = str(Path(model_dir).resolve())

    config = PipelineConfig(**overrides)
    config.ensure_dirs()
    ensure_captions(config.data_path)

    stages = list(payload.get("stages") or preset.get("stages") or PRESETS["smoke"]["stages"])
    for stage in stages:
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unknown pipeline stage: {stage}")

    return {
        "job_id": job_id,
        "user_id": user_id,
        "preset": preset_name,
        "output_root": str(output_root.resolve()),
        "stages": stages,
        "config": config,
    }
