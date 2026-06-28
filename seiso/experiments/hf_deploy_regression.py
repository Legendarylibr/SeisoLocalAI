"""Measure deployment-quant regression with real GPU eval on merged HF weights."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Any

from seiso.experiments._metrics import finite_float, finite_floats
from seiso.training.config import QuantMode, TrainConfig

DEFAULT_DEPLOY_QUANTS: tuple[str, ...] = ("4bit", "8bit", "16bit")

_DEPLOY_BITS: dict[str, float] = {
    "4bit": 4.5,
    "8bit": 8.5,
    "16bit": 16.0,
}


def _load_eval_texts(
    train_out: Path,
    base_config: TrainConfig,
    *,
    max_samples: int = 16,
) -> list[str]:
    snapshot = train_out / "train_config_snapshot.json"
    cfg = base_config
    if snapshot.is_file():
        cfg = TrainConfig.model_validate(
            json.loads(snapshot.read_text(encoding="utf-8"))
        )

    from seiso.training.datasets import format_dataset_text, load_training_dataset

    raw = load_training_dataset(cfg.dataset, sandbox_root=cfg.sandbox_root)
    max_total = cfg.extra.get("max_samples")
    if isinstance(max_total, int) and max_total > 0 and len(raw) > max_total:
        raw = raw.select(range(max_total))
    if cfg.eval_split_ratio > 0 and len(raw) > 10:
        split = raw.train_test_split(test_size=cfg.eval_split_ratio, seed=cfg.seed)
        eval_ds = split["test"]
    else:
        eval_ds = raw.select(range(min(max_samples, len(raw))))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(cfg.model_id), trust_remote_code=False
    )
    eval_ds, _ = format_dataset_text(eval_ds, tokenizer, cfg.dataset_format)
    texts = [str(text) for text in eval_ds["text"][:max_samples]]
    return [t for t in texts if t.strip()]


def _eval_merged_loss(
    merged_dir: Path,
    *,
    model_id: str,
    deploy_quant: str,
    texts: list[str],
    max_seq_length: int,
    on_log: Callable[[str], None] | None = None,
) -> tuple[float, float]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = QuantMode(deploy_quant)
    load_4bit = quant == QuantMode.INT4
    load_8bit = quant == QuantMode.INT8
    dtype = torch.float16 if quant == QuantMode.INT16 else None

    if on_log:
        on_log(
            f"Evaluating merged weights deploy_quant={deploy_quant} samples={len(texts)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir), trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"device_map": "auto", "low_cpu_mem_usage": True}
    if load_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(str(merged_dir), **model_kwargs)
    model.eval()

    losses: list[float] = []
    for text in texts:
        batch = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
        )
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.inference_mode():
            out = model(**batch, labels=batch["input_ids"])
            loss = float(out.loss.detach().cpu())
        if math.isfinite(loss):
            losses.append(loss)

    del model
    try:
        from seiso.memory.protection import release_cached_memory

        release_cached_memory()
    except ImportError:
        torch.cuda.empty_cache()

    if not losses:
        raise RuntimeError(f"No finite eval losses for deploy_quant={deploy_quant}")
    avg_loss = mean(losses)
    return avg_loss, math.exp(min(20.0, avg_loss))


def _select_lowest_memory_without_regression(
    rows: list[dict[str, Any]],
    *,
    max_reward_regression: float,
    max_perplexity_regression: float | None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No deploy-quant rows to compare")
    best_reward = max(float(row["reward"]) for row in rows)
    best_perplexity = min(float(row["perplexity"]) for row in rows)
    eligible: list[dict[str, Any]] = []
    for row in rows:
        reward_regression = best_reward - float(row["reward"])
        perplexity_regression = float(row["perplexity"]) - best_perplexity
        if reward_regression > max_reward_regression:
            continue
        if (
            max_perplexity_regression is not None
            and perplexity_regression > max_perplexity_regression
        ):
            continue
        eligible.append(
            {
                **row,
                "reward_regression": reward_regression,
                "perplexity_regression": perplexity_regression,
            }
        )
    if eligible:
        selected = min(eligible, key=lambda row: float(row["memory_mb"]))
        selected["reason"] = "lowest memory within regression bounds"
        return selected
    best = max(rows, key=lambda row: float(row["reward"]))
    return {
        **best,
        "reward_regression": 0.0,
        "perplexity_regression": 0.0,
        "reason": "no candidate met regression bounds; selected best reward",
    }


def run_hf_deploy_quant_regression(
    merged_dir: Path,
    *,
    train_out: Path,
    base_config: TrainConfig,
    deploy_quants: list[str] | tuple[str, ...] = DEFAULT_DEPLOY_QUANTS,
    max_eval_samples: int = 16,
    max_reward_regression: float = 0.05,
    max_perplexity_regression: float = 0.02,
    parameters_b: float | None = None,
    on_log: Callable[[str], None] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Compare deployment quants on merged HF weights using GPU eval loss/perplexity."""
    from seiso.models.hub_quant import infer_active_params_b

    params_b = (
        parameters_b
        if parameters_b is not None
        else infer_active_params_b(base_config.model_id)
    )
    texts = _load_eval_texts(train_out, base_config, max_samples=max_eval_samples)
    if not texts:
        raise RuntimeError(
            "No eval texts available — increase dataset size or eval_split_ratio"
        )

    rows: list[dict[str, Any]] = []
    for deploy_quant in deploy_quants:
        loss, perplexity = _eval_merged_loss(
            merged_dir,
            model_id=base_config.model_id,
            deploy_quant=deploy_quant,
            texts=texts,
            max_seq_length=min(base_config.max_seq_length, 1024),
            on_log=on_log,
        )
        bits = _DEPLOY_BITS.get(deploy_quant, 16.0)
        rows.append(
            {
                "route_id": f"hf_deploy_{deploy_quant}",
                "quant_label": deploy_quant.upper(),
                "deploy_quant": deploy_quant,
                "eval_loss": loss,
                "perplexity": perplexity,
                "reward": -loss,
                "memory_mb": params_b * 1000 * bits / 8,
            }
        )

    recommendations = [
        _select_lowest_memory_without_regression(
            rows,
            max_reward_regression=max_reward_regression,
            max_perplexity_regression=max_perplexity_regression,
        )
    ]
    report = {
        "measurement": "hf_merged_eval",
        "samples": len(rows) * len(texts),
        "prompt_count": len(texts),
        "route_count": len(rows),
        "max_reward_regression": max_reward_regression,
        "max_perplexity_regression": max_perplexity_regression,
        "rows": rows,
        "recommendations": recommendations,
        "mean_selected_memory_mb": recommendations[0].get("memory_mb"),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "hf_deploy_regression.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["artifact_path"] = str(path)
    return report


def summarize_hf_deploy_report(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("rows") if isinstance(report.get("rows"), list) else []
    recommendations = (
        report.get("recommendations")
        if isinstance(report.get("recommendations"), list)
        else []
    )
    selected = recommendations[0] if recommendations else {}
    rewards = finite_floats(
        [row.get("reward") for row in rows if isinstance(row, dict)]
    )
    perplexities = finite_floats(
        [row.get("perplexity") for row in rows if isinstance(row, dict)]
    )
    return {
        "eval_mean_reward": mean(rewards) if rewards else None,
        "eval_mean_perplexity": mean(perplexities) if perplexities else None,
        "recommended_route": selected.get("route_id"),
        "recommended_quant": selected.get("quant_label")
        or selected.get("deploy_quant"),
        "reward_regression": finite_float(selected.get("reward_regression")),
        "perplexity_regression": finite_float(selected.get("perplexity_regression")),
        "mean_selected_memory_mb": finite_float(selected.get("memory_mb")),
    }
