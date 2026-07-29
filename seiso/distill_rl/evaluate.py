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
    trust_remote_code: bool = False,
    use_chat_template: bool = False,
    benchmark_verifiable: bool = False,
    benchmark_tasks: list[str] | None = None,
    require_thinking_trace: bool = False,
    thinking_instruction: str = (
        "Show your reasoning in <think>...</think>, then give the final answer."
    ),
    on_log=None,
) -> dict[str, Any]:
    """Evaluate named checkpoints: perplexity, val preference accuracy, samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if prompt_library_path is not None:
        eval_prompts = load_rollout_prompts(prompt_library_path, limit=eval_max_prompts)
        eval_texts = [prompt.text for prompt in eval_prompts]
    else:
        # Prefer prompts embedded in val preferences when no library is configured
        # (data_designer Distill-RL default).
        val_probe = _load_jsonl(val_preferences_path)
        eval_texts = []
        for row in val_probe:
            prompt = str(row.get("prompt") or "").strip()
            if prompt and prompt not in eval_texts:
                eval_texts.append(prompt)
            if len(eval_texts) >= max(1, eval_max_prompts):
                break
        eval_prompts = [
            RolloutPrompt(prompt_id=f"pref_{idx}", text=text)
            for idx, text in enumerate(eval_texts)
        ]

    results: dict[str, Any] = {
        "checkpoints": {},
        "eval_prompt_count": len(eval_prompts),
        "use_chat_template": bool(use_chat_template),
    }
    val_rows = _load_jsonl(val_preferences_path)
    skipped: list[str] = []

    for name, model_ref in checkpoints.items():
        model_path = _resolve_model_ref(model_ref)
        if model_path is None:
            skipped.append(f"{name}:{model_ref!r}")
            continue
        if on_log:
            on_log(f"Evaluate: {name} → {model_path}")
        metrics = _evaluate_checkpoint(
            model_path,
            eval_texts,
            val_rows,
            trust_remote_code=trust_remote_code,
            use_chat_template=use_chat_template,
        )
        metrics["samples"] = _write_samples(
            output_dir / f"samples_{name}.jsonl",
            model_path,
            eval_prompts,
            trust_remote_code=trust_remote_code,
            use_chat_template=use_chat_template,
        )
        results["checkpoints"][name] = metrics

    if checkpoints and not results["checkpoints"]:
        detail = ", ".join(skipped) if skipped else "none resolved"
        raise FileNotFoundError(
            f"Evaluate found no usable checkpoints (skipped: {detail})"
        )
    if skipped and on_log:
        on_log(f"Evaluate skipped missing checkpoints: {', '.join(skipped)}")

    if benchmark_verifiable:
        from seiso.distill_rl.verifiable_benchmarks import (
            evaluate_verifiable_benchmarks,
        )

        benchmark_checkpoints = {
            name: str(_resolve_model_ref(model_ref))
            for name, model_ref in checkpoints.items()
            if _resolve_model_ref(model_ref) is not None
        }
        results["verifiable_benchmarks"] = evaluate_verifiable_benchmarks(
            output_dir=output_dir,
            checkpoints=benchmark_checkpoints,
            prompt_library_path=prompt_library_path,
            benchmark_tasks=benchmark_tasks or ["gsm8k", "gpqa", "aime"],
            max_prompts_per_task=max(1, eval_max_prompts),
            trust_remote_code=trust_remote_code,
            require_thinking_trace=require_thinking_trace,
            thinking_instruction=thinking_instruction,
            on_log=on_log,
        )

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
    *,
    trust_remote_code: bool = False,
    use_chat_template: bool = False,
) -> dict[str, Any]:
    from seiso.compress.bootstrap import require_codellama_compress

    require_codellama_compress()
    from seiso.codellama_compress.evaluate import compute_perplexity, measure_speed

    model, tokenizer, device = load_causal_lm(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    try:
        ppl = compute_perplexity(model, tokenizer, eval_texts, device)
        tps, ms = measure_speed(
            model,
            tokenizer,
            eval_texts[0] if eval_texts else "def fib(n):",
            device,
        )
        val_metrics = _val_preference_metrics(
            model,
            tokenizer,
            val_rows,
            device,
            use_chat_template=use_chat_template,
        )
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
    *,
    use_chat_template: bool = False,
) -> dict[str, float | int | bool | None]:
    if not val_rows:
        return {
            "val_preference_accuracy": None,
            "val_preference_margin_mean": None,
            "val_preference_margin_median": None,
            "val_preference_count": 0,
            "alignment_score": None,
            "val_preference_unavailable": True,
        }

    correct = 0
    margins: list[float] = []
    for row in val_rows:
        # Match DPO training: chat-template formatting + joint tokenize + sum.
        chosen_lp = _sequence_logprob(
            model,
            tokenizer,
            row["prompt"],
            row["chosen"],
            device,
            average=False,
            use_chat_template=use_chat_template,
        )
        rejected_lp = _sequence_logprob(
            model,
            tokenizer,
            row["prompt"],
            row["rejected"],
            device,
            average=False,
            use_chat_template=use_chat_template,
        )
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


def _format_eval_prompt(tokenizer, prompt: str, *, use_chat_template: bool) -> str:
    """Mirror DPODataCollator._format_prompt for val metrics."""
    if not use_chat_template:
        return prompt
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_template is None:
        return prompt
    return apply_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _sequence_logprob(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    device,
    *,
    average: bool = False,
    use_chat_template: bool = False,
) -> float:
    """Completion logprob with joint tokenization (matches DPO collator).

    Prompt length is derived from a joint encode of ``prompt+completion`` so
    BPE merges across the boundary cannot empty the completion label span.
    When ``use_chat_template`` is set, format the prompt the same way as training.
    """
    import torch

    prompt_text = _format_eval_prompt(
        tokenizer, prompt, use_chat_template=use_chat_template
    )
    add_special = not use_chat_template
    text = f"{prompt_text}{completion}"
    joint = tokenizer(text, return_tensors="pt", add_special_tokens=add_special)
    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=add_special
    )["input_ids"][0]
    prompt_len = min(int(prompt_ids.numel()), int(joint["input_ids"].shape[1]) - 1)
    if prompt_len < 1:
        prompt_len = 1
    enc = {k: v.to(device) for k, v in joint.items()}
    with torch.inference_mode():
        out = model(**enc)
        logits = out.logits[0, prompt_len - 1 : -1]
        labels = enc["input_ids"][0, prompt_len:]
        if labels.numel() == 0:
            return 0.0
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        if average:
            return float(token_log_probs.mean().item())
        return float(token_log_probs.sum().item())


def _write_samples(
    path: Path,
    model_path: str,
    prompts: list[RolloutPrompt],
    *,
    trust_remote_code: bool = False,
    use_chat_template: bool = False,
) -> str:
    from seiso.distill_rl.rollouts import generate_completions

    outputs = generate_completions(
        model_path,
        prompts,
        max_new_tokens=64,
        temperature=0.0,
        seed=0,
        use_chat_template=use_chat_template,
        trust_remote_code=trust_remote_code,
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
