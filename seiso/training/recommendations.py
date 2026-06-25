"""Hardware- and model-aware training configuration recommendations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from seiso.hardware.training import training_defaults
from seiso.memory.estimates import estimate_training_vram_gb, guess_params_from_name
from seiso.models.catalog import _parse_param_size
from seiso.models.trainable_snapshot import GGUF_ONLY_REPO_MESSAGE, is_gguf_only_repo_id
from seiso.training.config import DatasetFormat
from seiso.training.dataset_analysis import (
    analysis_notes_for_recommendations,
    analyze_training_dataset,
)

logger = logging.getLogger(__name__)

_FALLBACK_TRAIN_REPO = "Qwen/Qwen2.5-0.5B-Instruct"

_DATASET_HINTS: dict[str, dict[str, Any]] = {
    "huggingfaceh4/no_robots": {
        "dataset_format": "chat",
        "train_on_responses_only": True,
        "note": "Chat-style preference data — use chat format and train on assistant replies only.",
    },
    "tatsu-lab/alpaca": {
        "dataset_format": "alpaca",
        "train_on_responses_only": False,
        "note": "Instruction/output pairs — Alpaca format works best.",
    },
    "yahma/alpaca-cleaned": {
        "dataset_format": "alpaca",
        "train_on_responses_only": False,
        "note": "Classic instruction tuning dataset — Alpaca format.",
    },
    "databricks/databricks-dolly-15k": {
        "dataset_format": "alpaca",
        "train_on_responses_only": False,
        "note": "Instruction/context/response columns — Alpaca-style mapping.",
    },
    "open-orca/openorca": {
        "dataset_format": "sharegpt",
        "train_on_responses_only": True,
        "note": "Multi-turn conversations — ShareGPT or chat format.",
    },
    "meta-math/metamathqa": {
        "dataset_format": "alpaca",
        "train_on_responses_only": True,
        "note": "Math Q&A pairs — uses query/response columns with assistant-only loss.",
    },
    "openai/gsm8k": {
        "dataset_format": "alpaca",
        "train_on_responses_only": True,
        "note": "Grade-school math word problems — question/answer columns.",
    },
    "lighteval/math": {
        "dataset_format": "alpaca",
        "train_on_responses_only": True,
        "note": "Competition math problems — question/solution columns when present.",
    },
}


def _model_params_b(model_id: str) -> float | None:
    guessed = guess_params_from_name(model_id)
    if guessed is not None:
        return guessed
    try:
        raw = _parse_param_size(model_id.split("/")[-1])
    except ValueError:
        return None
    return raw if raw != float("inf") else None


def _dataset_hints(dataset: str) -> dict[str, Any]:
    key = dataset.strip().lower()
    if key in _DATASET_HINTS:
        return dict(_DATASET_HINTS[key])
    if "alpaca" in key:
        return {
            "dataset_format": "alpaca",
            "train_on_responses_only": False,
            "note": "Name suggests instruction tuning — Alpaca format is a safe default.",
        }
    if "preference" in key:
        return {
            "dataset_format": "preference",
            "train_on_responses_only": True,
            "note": "Preference data — uses chosen responses parsed as chat turns.",
        }
    if any(token in key for token in ("chat", "conversation", "sharegpt", "messages")):
        return {
            "dataset_format": "chat",
            "train_on_responses_only": True,
            "note": "Name suggests chat data — chat format with response-only loss.",
        }
    if any(token in key for token in ("code", "pretrain", "pretraining")):
        return {
            "dataset_format": "text",
            "train_on_responses_only": False,
            "note": "Code/pretraining corpus — normalize to a single text field (see scripts/prepare_code_corpus.py).",
        }
    return {
        "dataset_format": "auto",
        "train_on_responses_only": True,
        "note": "Leave format on Auto-detect unless you know the schema.",
    }


def _scale_for_model_size(base: dict[str, Any], params_b: float | None) -> dict[str, Any]:
    cfg = dict(base)
    if params_b is None:
        return cfg

    if params_b <= 1.0:
        cfg["batch_size"] = min(int(cfg["batch_size"]), 4)
        cfg["gradient_accumulation_steps"] = max(2, int(cfg["gradient_accumulation_steps"]) // 2)
        cfg["lora_r"] = 16
        cfg["lora_alpha"] = 32
        cfg["epochs"] = 5
    elif params_b <= 3.0:
        cfg["batch_size"] = min(int(cfg["batch_size"]), 2)
        cfg["lora_r"] = 16
        cfg["lora_alpha"] = 32
        cfg["epochs"] = 3
    elif params_b <= 7.0:
        cfg["batch_size"] = 1
        cfg["gradient_accumulation_steps"] = max(int(cfg["gradient_accumulation_steps"]), 8)
        cfg["max_seq_length"] = min(int(cfg["max_seq_length"]), 2048)
        cfg["lora_r"] = 16
        cfg["lora_alpha"] = 32
    elif params_b <= 14.0:
        cfg["batch_size"] = 1
        cfg["gradient_accumulation_steps"] = max(int(cfg["gradient_accumulation_steps"]), 16)
        cfg["max_seq_length"] = min(int(cfg["max_seq_length"]), 2048)
        cfg["lora_r"] = 8
        cfg["lora_alpha"] = 16
        cfg["quant"] = "4bit"
    else:
        cfg["batch_size"] = 1
        cfg["gradient_accumulation_steps"] = max(int(cfg["gradient_accumulation_steps"]), 32)
        cfg["max_seq_length"] = min(int(cfg["max_seq_length"]), 1024)
        cfg["lora_r"] = 8
        cfg["lora_alpha"] = 16
        cfg["quant"] = "4bit"

    return cfg


def _try_analyze_dataset(
    dataset: str,
    *,
    sandbox_root: Path | None = None,
) -> dict[str, Any] | None:
    if not dataset.strip():
        return None
    try:
        return analyze_training_dataset(
            dataset,
            dataset_format=DatasetFormat.AUTO,
            sandbox_root=sandbox_root,
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
) -> dict[str, Any]:
    """Return suggested training knobs for the current hardware, model, and dataset."""
    defaults = training_defaults(profile)
    params_b = _model_params_b(model_id)
    analysis = _try_analyze_dataset(dataset, sandbox_root=sandbox_root)
    ds = (
        analysis.get("recommended_config", {})
        if analysis
        else _dataset_hints(dataset)
    )
    warnings: list[str] = []
    notes: list[str] = [defaults["note"]]

    trainable = not is_gguf_only_repo_id(model_id)
    if model_id and not trainable:
        warnings.append(GGUF_ONLY_REPO_MESSAGE)

    base_cfg = {
        "method": defaults["method"],
        "quant": defaults["quant"],
        "batch_size": defaults["batch_size"],
        "gradient_accumulation_steps": defaults["gradient_accumulation_steps"],
        "max_seq_length": ds.get("max_seq_length", defaults["max_seq_length"]),
        "learning_rate": 2e-4,
        "epochs": ds.get("epochs", 5),
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
        "max_eval_samples": 128,
        "early_stopping": ds.get("early_stopping", True),
        "early_stopping_patience": ds.get("early_stopping_patience", 3),
    }
    config = _scale_for_model_size(base_cfg, params_b)

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
    elif ds.get("note"):
        notes.append(str(ds["note"]))

    est_vram_gb = None
    if params_b is not None:
        est_vram_gb = estimate_training_vram_gb(
            f"{params_b}B",
            quant=str(config["quant"]),
            repo_id=model_id,
        )

    payload: dict[str, Any] = {
        "config": config,
        "warnings": warnings,
        "notes": notes,
        "trainable": trainable,
        "model_params": f"{params_b:g}B" if params_b is not None else None,
        "est_training_vram_gb": est_vram_gb,
        "fallback_train_repo": _FALLBACK_TRAIN_REPO,
        "hardware_tier": profile.get("tier_label") or profile.get("tier"),
    }
    if analysis:
        payload["dataset_analysis"] = analysis
    return payload
