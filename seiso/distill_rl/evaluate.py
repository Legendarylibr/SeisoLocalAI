"""Evaluate checkpoints for research reporting."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from seiso.distill_rl.model_utils import load_causal_lm, release_causal_lm
from seiso.distill_rl.prompts import RolloutPrompt, load_rollout_prompts


def evaluate_pipeline(
    *,
    output_dir: Path,
    checkpoints: dict[str, Path | str],
    val_preferences_path: Path,
    prompt_library_path: Path | None,
    eval_max_prompts: int,
    on_log=None,
) -> dict[str, Any]:
    """Evaluate named checkpoints: perplexity, val preference accuracy, samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_prompts = load_rollout_prompts(prompt_library_path, limit=eval_max_prompts)
    eval_texts = [prompt.text for prompt in eval_prompts]

    results: dict[str, Any] = {"checkpoints": {}, "eval_prompt_count": len(eval_prompts)}
    val_rows = _load_jsonl(val_preferences_path)

    for name, model_ref in checkpoints.items():
        model_path = _resolve_model_ref(model_ref)
        if model_path is None:
            continue
        if on_log:
            on_log(f"Evaluate: {name} → {model_path}")
        metrics = _evaluate_checkpoint(model_path, eval_texts, val_rows)
        metrics["samples"] = _write_samples(
            output_dir / f"samples_{name}.jsonl",
            model_path,
            eval_prompts,
        )
        results["checkpoints"][name] = metrics

    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    results["summary_path"] = str(summary_path)
    return results


def _resolve_model_ref(model_ref: Path | str) -> str | None:
    if isinstance(model_ref, Path):
        return str(model_ref.resolve()) if model_ref.is_dir() else None
    text = str(model_ref).strip()
    return text or None


def _evaluate_checkpoint(
    model_path: str,
    eval_texts: list[str],
    val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    from seiso.compress.bootstrap import require_codellama_compress

    require_codellama_compress()
    from codellama_compress.evaluate import compute_perplexity, measure_speed

    model, tokenizer, device = load_causal_lm(model_path)
    try:
        ppl = compute_perplexity(model, tokenizer, eval_texts, device)
        tps, ms = measure_speed(
            model,
            tokenizer,
            eval_texts[0] if eval_texts else "def fib(n):",
            device,
        )
        val_metrics = _val_preference_metrics(model, tokenizer, val_rows, device)
    finally:
        release_causal_lm(model)

    return {
        "model_dir": model_path,
        "perplexity": ppl,
        "tokens_per_second": tps,
        "avg_time_ms": ms,
        **val_metrics,
    }


def _val_preference_metrics(
    model,
    tokenizer,
    val_rows: list[dict[str, Any]],
    device,
) -> dict[str, float | int]:
    if not val_rows:
        return {
            "val_preference_accuracy": 0.0,
            "val_preference_margin_mean": 0.0,
            "val_preference_margin_median": 0.0,
            "val_preference_count": 0,
            "alignment_score": 0.0,
        }

    correct = 0
    margins: list[float] = []
    for row in val_rows:
        chosen_lp = _sequence_logprob(model, tokenizer, row["prompt"], row["chosen"], device)
        rejected_lp = _sequence_logprob(model, tokenizer, row["prompt"], row["rejected"], device)
        margin = chosen_lp - rejected_lp
        margins.append(margin)
        if margin > 0:
            correct += 1
    accuracy = correct / len(val_rows)
    mean_margin = statistics.fmean(margins)
    median_margin = statistics.median(margins)
    alignment_score = accuracy + 0.05 * math.tanh(mean_margin)
    return {
        "val_preference_accuracy": accuracy,
        "val_preference_margin_mean": mean_margin,
        "val_preference_margin_median": median_margin,
        "val_preference_count": len(val_rows),
        "alignment_score": alignment_score,
    }


def _sequence_logprob(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    device,
    *,
    average: bool = True,
) -> float:
    import torch

    text = f"{prompt}{completion}"
    enc = tokenizer(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_len = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
    with torch.inference_mode():
        out = model(**enc)
        logits = out.logits[0, prompt_len - 1 : -1]
        labels = enc["input_ids"][0, prompt_len:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        if average:
            return float(token_log_probs.mean().item()) if token_log_probs.numel() else 0.0
        return float(token_log_probs.sum().item())


def _write_samples(path: Path, model_path: str, prompts: list[RolloutPrompt]) -> str:
    from seiso.distill_rl.rollouts import generate_completions

    outputs = generate_completions(
        model_path,
        prompts,
        max_new_tokens=64,
        temperature=0.0,
        seed=0,
        use_chat_template=False,
    )
    with path.open("w", encoding="utf-8") as handle:
        for prompt, completion in zip(prompts, outputs, strict=True):
            handle.write(
                json.dumps(
                    {
                        "prompt_id": prompt.prompt_id,
                        "prompt": prompt.text,
                        "completion": completion,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return str(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
