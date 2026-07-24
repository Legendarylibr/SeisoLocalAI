"""Hardware- and dataset-aware training configuration recommendations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from seiso.hardware.training import training_defaults
from seiso.memory.estimates import estimate_training_vram_gb, guess_params_from_name
from seiso.models.catalog import _parse_param_size
from seiso.models.hub_quant import (
    infer_moe_sizing,
    native_quant_training_block_reason,
)
from seiso.models.trainable_snapshot import GGUF_ONLY_REPO_MESSAGE, is_gguf_only_repo_id
from seiso.training.config import DatasetFormat, TrainMethod
from seiso.training.dataset_analysis import (
    analysis_notes_for_recommendations,
    analyze_training_dataset,
)
from seiso.training.practices import learning_rate_for_method, warmup_ratio_for_corpus

logger = logging.getLogger(__name__)


def _default_dataset_config() -> dict[str, Any]:
    """Schema-agnostic defaults when no dataset analysis is available."""
    return {
        "dataset_format": "auto",
        "train_on_responses_only": True,
        "preprocess_dataset": True,
        "deduplicate_dataset": True,
        "packing": False,
        "preference_as_sft": False,
        "early_stopping": True,
        "early_stopping_patience": 3,
    }


def _dataset_format_hint(dataset: str) -> str:
    """Infer common dataset families when full dataset analysis is unavailable."""
    name = dataset.strip().lower()
    if not name:
        return "auto"
    if "alpaca" in name:
        return "alpaca"
    if any(marker in name for marker in ("chat", "sharegpt", "ultrachat", "no_robots")):
        return "chat"
    if any(
        marker in name for marker in ("preference", "dpo", "orpo", "chosen", "rejected")
    ):
        return "preference"
    return "auto"


def _model_params_b(model_id: str) -> float | None:
    if not model_id.strip():
        return None
    guessed = guess_params_from_name(model_id)
    if guessed is not None:
        return guessed
    try:
        raw = _parse_param_size(model_id.split("/")[-1])
    except ValueError:
        return None
    return raw if raw != float("inf") else None


def _apply_hardware_caps(
    base: dict[str, Any],
    params_b: float | None,
    *,
    hardware_max_seq: int,
) -> dict[str, Any]:
    """Cap memory-sensitive knobs from hardware/parameter count without overriding dataset choices."""
    cfg = dict(base)
    if params_b is None:
        return cfg

    if params_b <= 1.0:
        cfg["batch_size"] = min(int(cfg.get("batch_size", 1)), 4)
        cfg["gradient_accumulation_steps"] = max(
            2, int(cfg.get("gradient_accumulation_steps", 4)) // 2
        )
        cfg.setdefault("lora_r", 16)
        cfg.setdefault("lora_alpha", 32)
    elif params_b <= 3.0:
        cfg["batch_size"] = min(int(cfg.get("batch_size", 1)), 2)
        cfg.setdefault("lora_r", 16)
        cfg.setdefault("lora_alpha", 32)
    elif params_b <= 7.0:
        cfg["batch_size"] = min(int(cfg.get("batch_size", 1)), 1)
        cfg["gradient_accumulation_steps"] = max(
            int(cfg.get("gradient_accumulation_steps", 4)), 8
        )
        cfg.setdefault("lora_r", 16)
        cfg.setdefault("lora_alpha", 32)
    elif params_b <= 14.0:
        cfg["batch_size"] = 1
        cfg["gradient_accumulation_steps"] = max(
            int(cfg.get("gradient_accumulation_steps", 4)), 16
        )
        cfg.setdefault("lora_r", 8)
        cfg.setdefault("lora_alpha", 16)
        cfg.setdefault("quant", "4bit")
    else:
        cfg["batch_size"] = 1
        cfg["gradient_accumulation_steps"] = max(
            int(cfg.get("gradient_accumulation_steps", 4)), 32
        )
        cfg.setdefault("lora_r", 8)
        cfg.setdefault("lora_alpha", 16)
        cfg.setdefault("quant", "4bit")

    seq_cap = hardware_max_seq
    if params_b > 14.0:
        seq_cap = min(seq_cap, 1024)
    elif params_b > 3.0:
        seq_cap = min(seq_cap, 2048)
    cfg["max_seq_length"] = min(int(cfg.get("max_seq_length", seq_cap)), seq_cap)

    return cfg


def _try_analyze_dataset(
    dataset: str,
    *,
    sandbox_root: Path | None = None,
    sandbox_user_id: str | None = None,
) -> dict[str, Any] | None:
    if not dataset.strip():
        return None
    try:
        return analyze_training_dataset(
            dataset,
            dataset_format=DatasetFormat.AUTO,
            sandbox_root=sandbox_root,
            sandbox_user_id=sandbox_user_id,
        )
    except Exception as exc:
        logger.info("Dataset analysis unavailable for %s: %s", dataset, exc)
        return None


def recommend_training_config(
    profile: dict[str, Any],
    *,
    model_id: str = "",
    dataset: str = "",
    sandbox_root: Path | None = None,
    sandbox_user_id: str | None = None,
) -> dict[str, Any]:
    """Return suggested training knobs for the current hardware, model, and dataset."""
    defaults = training_defaults(profile)
    params_b = _model_params_b(model_id)
    moe_sizing = infer_moe_sizing(model_id) if model_id else None
    if moe_sizing and moe_sizing.total_params_b is not None:
        params_b = moe_sizing.total_params_b
    analysis = _try_analyze_dataset(
        dataset, sandbox_root=sandbox_root, sandbox_user_id=sandbox_user_id
    )
    ds = (
        analysis.get("recommended_config", {})
        if analysis
        else _default_dataset_config()
    )
    if not analysis and ds.get("dataset_format", "auto") == "auto":
        ds["dataset_format"] = _dataset_format_hint(dataset)
    warnings: list[str] = []
    notes: list[str] = [defaults["note"]]

    trainable = not model_id or not is_gguf_only_repo_id(model_id)
    if model_id and not trainable:
        warnings.append(GGUF_ONLY_REPO_MESSAGE)
    native_quant_block = (
        native_quant_training_block_reason(model_id) if model_id else None
    )
    if native_quant_block:
        trainable = False
        warnings.append(native_quant_block)

    base_cfg = {
        "method": defaults["method"],
        "quant": defaults["quant"],
        "batch_size": defaults["batch_size"],
        "gradient_accumulation_steps": defaults["gradient_accumulation_steps"],
        "max_seq_length": ds.get("max_seq_length", defaults["max_seq_length"]),
        "learning_rate": ds.get(
            "learning_rate",
            learning_rate_for_method(TrainMethod(defaults["method"])),
        ),
        "warmup_ratio": ds.get(
            "warmup_ratio",
            warmup_ratio_for_corpus(int(analysis.get("kept", 0)) if analysis else 0),
        ),
        "epochs": ds.get("epochs", 3),
        "lora_r": 16,
        "lora_alpha": 32,
        "gradient_checkpointing": defaults["gradient_checkpointing"],
        "use_triton": defaults.get("use_fused_kernels", True),
        "use_fused_ce": defaults.get("use_fused_ce", True),
        "train_on_responses_only": ds.get("train_on_responses_only", True),
        "use_rslora": False,
        "packing": ds.get("packing", False),
        "dataset_format": ds.get("dataset_format", "auto"),
        "preprocess_dataset": ds.get("preprocess_dataset", True),
        "deduplicate_dataset": ds.get("deduplicate_dataset", True),
        "preference_as_sft": ds.get("preference_as_sft", False),
        "neftune_noise_alpha": ds.get("neftune_noise_alpha", 5.0),
        "max_eval_samples": 128,
        "early_stopping": ds.get("early_stopping", True),
        "early_stopping_patience": ds.get("early_stopping_patience", 3),
    }
    # MoE caps must use resident totals (params_b), not active-per-token size.
    config = _apply_hardware_caps(
        base_cfg,
        params_b,
        hardware_max_seq=int(defaults["max_seq_length"]),
    )
    if (
        config["packing"]
        and config["train_on_responses_only"]
        and str(config.get("dataset_format", "auto")) != "text"
    ):
        config["packing"] = False
    if str(config.get("dataset_format", "")) == "preference":
        config["preference_as_sft"] = False
        config["packing"] = False
        warnings.append(
            "Preference pairs are not SFT alignment — use Distill-RL/DPO "
            "(`seiso distill-rl`), or set preference_as_sft=true for chosen-only SFT."
        )
    if moe_sizing and moe_sizing.is_moe:
        config.update(
            {
                "batch_size": min(int(config["batch_size"]), 1),
                "lora_r": min(int(config["lora_r"]), 16),
                "lora_alpha": min(int(config["lora_alpha"]), 32),
                "gradient_checkpointing": True,
            }
        )
        notes.append(
            "MoE fine-tune is optional: enable MoE-aware LoRA to freeze the router "
            "by default and reduce routing instability."
        )

    if params_b is not None:
        max_rec = defaults.get("max_recommended_params", "7B")
        try:
            max_b = _parse_param_size(max_rec)
        except ValueError:
            max_b = 7.0
        if params_b > max_b * 1.05:
            warnings.append(
                f"Model is ~{params_b:g}B — your GPU profile recommends ≤{max_rec}. "
                "Expect tight VRAM; reduce max seq length or pick a smaller base."
            )
        notes.append(
            f"Base model ~{params_b:g}B — batch {config['batch_size']}, "
            f"grad accum {config['gradient_accumulation_steps']}, "
            f"seq {config['max_seq_length']}."
        )

    if analysis:
        notes.extend(analysis_notes_for_recommendations(analysis))

    est_vram_gb = None
    if params_b is not None:
        est_vram_gb = estimate_training_vram_gb(
            f"{params_b}B",
            quant=str(config["quant"]),
            repo_id="" if moe_sizing and moe_sizing.is_moe else model_id,
        )

    payload: dict[str, Any] = {
        "config": config,
        "warnings": warnings,
        "notes": notes,
        "trainable": trainable,
        "model_params": f"{params_b:g}B" if params_b is not None else None,
        "est_training_vram_gb": est_vram_gb,
        "hardware_tier": profile.get("tier_label") or profile.get("tier"),
        "is_moe": bool(moe_sizing and moe_sizing.is_moe),
        "total_params_b": moe_sizing.total_params_b if moe_sizing else None,
        "active_params_b": moe_sizing.active_params_b if moe_sizing else None,
    }
    if analysis:
        payload["dataset_analysis"] = analysis
    return payload
