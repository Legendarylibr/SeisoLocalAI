"""Run vendored Stable Diffusion compression pipeline stages."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from seiso.image_compress.bootstrap import require_sd_compress
from seiso.image_compress.config_builder import build_pipeline_config


def _latest_model_dir(config: Any) -> str:
    for attr in ("quant_dir", "finetune_dir", "prune_dir", "distill_dir"):
        path = Path(getattr(config, attr))
        if path.is_dir() and any(path.iterdir()):
            return str(path)
    raise FileNotFoundError("No model artifacts found in run output")


def run_image_compress_job(
    *,
    job_id: str,
    user_id: str,
    data_dir: Path,
    payload: dict[str, Any],
    on_log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Execute baseline → distill → prune → finetune → quant (configurable)."""
    require_sd_compress()
    from sd_compress.baseline import generate_baseline
    from sd_compress.distillation import (
        cfg_distillation,
        clip_text_distillation,
        progressive_distillation,
    )
    from sd_compress.eval_runner import evaluate_stage, write_full_report
    from sd_compress.export import export_onnx, shard_safetensors
    from sd_compress.finetune import finetune_after_pruning
    from sd_compress.lora import test_lora_compatibility
    from sd_compress.optimization import install_runtime_helpers
    from sd_compress.pruning import prune_all
    from sd_compress.quantization import quantize
    from sd_compress.utils import LOGGER, configure_logging, report_torch_environment

    cfg_blob = build_pipeline_config(
        job_id=job_id,
        user_id=user_id,
        data_dir=data_dir,
        payload=payload,
    )
    config = cfg_blob["config"]
    stages = cfg_blob["stages"]

    configure_logging(str(payload.get("log_level", "INFO")))

    def _log(msg: str) -> None:
        LOGGER.info(msg)
        if on_log:
            on_log(msg)

    config.dump()
    env = report_torch_environment()
    _log(f"Image compression run preset={cfg_blob['preset']} stages={','.join(stages)}")
    _log(f"Environment: {env}")

    stage_results: dict[str, Any] = {}

    for stage in stages:
        _log(f"Phase: {stage}")

        if stage == "baseline":
            generate_baseline(config)
            stage_results["baseline"] = str(Path(config.eval_dir) / "baseline")

        elif stage == "distill_progressive":
            progressive_distillation(config)
            stage_results["distilled"] = config.distill_dir

        elif stage == "distill_clip":
            clip_text_distillation(config)
            stage_results["distilled"] = config.distill_dir

        elif stage == "distill_cfg":
            cfg_distillation(config)
            stage_results["distilled"] = config.distill_dir

        elif stage == "evaluate_distilled":
            result = evaluate_stage(config, "distilled", config.distill_dir)
            stage_results["evaluate_distilled"] = result

        elif stage == "prune":
            prune_all(config)
            stage_results["pruned"] = config.prune_dir

        elif stage == "evaluate_pruned":
            result = evaluate_stage(
                config,
                "pruned",
                config.prune_dir,
                compute_model_size=True,
            )
            stage_results["evaluate_pruned"] = result

        elif stage == "finetune":
            finetune_after_pruning(config)
            stage_results["finetuned"] = config.finetune_dir

        elif stage == "evaluate_finetuned":
            result = evaluate_stage(config, "finetuned", config.finetune_dir)
            stage_results["evaluate_finetuned"] = result

        elif stage == "quantize":
            quantize(config)
            stage_results["quantized"] = config.quant_dir

        elif stage == "evaluate_quantized":
            result = evaluate_stage(
                config,
                "quantized",
                config.quant_dir,
                compute_model_size=True,
            )
            stage_results["evaluate_quantized"] = result

        elif stage == "optimize":
            install_runtime_helpers(config)
            stage_results["optimized"] = config.quant_dir

        elif stage == "export_onnx":
            export_onnx(config)
            stage_results["export_onnx"] = config.export_dir

        elif stage == "export_shard":
            shard_safetensors(config)
            stage_results["export_shard"] = config.export_dir

        elif stage == "lora_test":
            test_lora_compatibility(config)
            stage_results["lora_test"] = config.quant_dir

        elif stage == "report":
            write_full_report(config)
            stage_results["report"] = str(Path(config.eval_dir) / "full_report.json")

        else:
            raise ValueError(f"Unhandled stage: {stage}")

    model_dir = (
        stage_results.get("quantized")
        or stage_results.get("finetuned")
        or stage_results.get("pruned")
        or stage_results.get("distilled")
        or _latest_model_dir(config)
    )

    _log("Image compression pipeline complete")

    return {
        "run_dir": str(Path(config.output_dir).resolve()),
        "output_root": cfg_blob["output_root"],
        "stages": stages,
        "stage_results": stage_results,
        "model_dir": model_dir,
    }
