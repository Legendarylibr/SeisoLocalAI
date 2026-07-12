#!/usr/bin/env python3
"""Benchmark a base or slime-LoRA checkpoint on competitive-coding unit tests."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _format_prompt(
    prompt: str,
    *,
    tokenizer,
    require_thinking: bool,
    instruction: str,
) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return tokenizer.apply_chat_template(
                messages,
                enable_thinking=require_thinking,
                **kwargs,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)
    if not require_thinking:
        return prompt
    if "<think>" in prompt.lower():
        return prompt
    return f"{prompt.rstrip()}\n\n{instruction}\n<think>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Optional PEFT adapter dir (slime checkpoint)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/nemotron_coding_bench.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-prompt-tokens", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--require-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Qwen3-style thinking (slower; default off for coding unit tests)",
    )
    parser.add_argument(
        "--thinking-instruction",
        default=(
            "Think step by step inside <think>...</think>, then provide the final "
            "Python solution in a single ```python code block that reads from stdin "
            "and writes to stdout."
        ),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/bench/coding_bench_report.json"),
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from seiso.slime_single_gpu.rewards import (
        extract_python_code,
        infer_reward_name,
        multi_reward,
        unit_tests_reward,
        codebase_tests_reward,
        uses_unit_tests_scoring,
    )

    rows = _load_jsonl(args.dataset, args.limit)
    if not rows:
        print(f"no rows in {args.dataset}", file=sys.stderr)
        return 1
    # Auto multi-reward when bench has mixed domains / explicit reward names.
    multi_mode = any(
        str(r.get("reward_name") or r.get("domain") or "").lower()
        in {"math", "multi", "auto", "mixed"}
        or (
            r.get("domain") == "math"
            or (r.get("answer") and not (r.get("unit_tests") or {}).get("inputs"))
        )
        for r in rows[: min(32, len(rows))]
    )
    # Prefer multi when any row is non-coding.
    multi_mode = multi_mode or any(
        infer_reward_name(r) != "unit_tests" for r in rows[: min(64, len(rows))]
    )
    if multi_mode:
        print("Benchmark mode: multi-reward (math + coding)", file=sys.stderr)
    else:
        print("Benchmark mode: unit_tests (coding)", file=sys.stderr)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"Loading model {args.model_id} dtype={args.dtype}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map={"": args.device},
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    if args.adapter is not None:
        from peft import PeftModel

        print(f"Loading adapter {args.adapter}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    results: list[dict[str, Any]] = []
    pass_rates: list[float] = []
    has_code_flags: list[bool] = []
    t0 = time.time()

    for i, sample in enumerate(rows):
        prompt = _format_prompt(
            str(sample["prompt"]),
            tokenizer=tokenizer,
            require_thinking=args.require_thinking,
            instruction=args.thinking_instruction,
        )
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_prompt_tokens,
        )
        encoded = {k: v.to(args.device) for k, v in encoded.items()}
        with torch.inference_mode():
            out = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = int(encoded["input_ids"].shape[1])
        completion = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)

        # Prefer answer after thinking; fall back to full completion for code fences.
        import re

        m = re.search(
            r"</think>(?P<final>.*)",
            completion,
            flags=re.IGNORECASE | re.DOTALL,
        )
        final = m.group("final").strip() if m else completion
        reward_name = infer_reward_name(sample) if multi_mode else (
            "codebase_tests"
            if (sample.get("codebase") or sample.get("reward_name") == "codebase_tests")
            else "unit_tests"
        )
        if multi_mode or reward_name == "codebase_tests":
            score = float(
                max(
                    multi_reward(final, sample),
                    multi_reward(completion, sample),
                )
            )
        else:
            score = float(
                max(
                    unit_tests_reward(final, sample),
                    unit_tests_reward(completion, sample),
                )
            )

        has_code = bool(extract_python_code(final) or extract_python_code(completion))
        if reward_name == "codebase_tests":
            # Count path-tagged or plain fences as code present.
            has_code = has_code or ("```" in completion and "def " in completion)
        # For math rows, treat "has_code" as non-blocking for quality gates:
        # mark True so code_extract_rate is not dragged down by pure-math benches.
        if reward_name == "math":
            has_code_flags.append(True)
        else:
            has_code_flags.append(has_code)
        pass_rates.append(score)
        row_out = {
            "index": i,
            "hash_id": sample.get("hash_id"),
            "source": sample.get("source"),
            "domain": sample.get("domain"),
            "reward_name": reward_name,
            "pass_rate": score,
            "has_code": has_code,
            "completion_chars": len(completion),
            "completion_preview": completion[:400],
        }
        results.append(row_out)
        print(
            f"[{i+1}/{len(rows)}] pass={score:.2f} reward={reward_name} "
            f"code={has_code} chars={len(completion)} source={sample.get('source')}",
            flush=True,
        )
        del out, encoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    mean_pass = sum(pass_rates) / len(pass_rates)
    perfect = sum(1 for x in pass_rates if x >= 0.999) / len(pass_rates)
    any_pass = sum(1 for x in pass_rates if x > 0) / len(pass_rates)
    code_rate = sum(1 for x in has_code_flags if x) / len(has_code_flags)

    math_scores = [r["pass_rate"] for r in results if r.get("reward_name") == "math"]
    code_scores = [r["pass_rate"] for r in results if r.get("reward_name") == "unit_tests"]
    report = {
        "model_id": args.model_id,
        "adapter": str(args.adapter) if args.adapter else None,
        "dataset": str(args.dataset),
        "mode": "multi" if multi_mode else "unit_tests",
        "n": len(rows),
        "mean_pass_rate": mean_pass,
        "perfect_rate": perfect,
        "any_pass_rate": any_pass,
        "code_extract_rate": code_rate,
        "math_mean": (sum(math_scores) / len(math_scores)) if math_scores else None,
        "coding_mean": (sum(code_scores) / len(code_scores)) if code_scores else None,
        "n_math": len(math_scores),
        "n_coding": len(code_scores),
        "elapsed_sec": elapsed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "mean_pass_rate": mean_pass,
                "perfect_rate": perfect,
                "any_pass_rate": any_pass,
                "code_extract_rate": code_rate,
                "n": len(rows),
                "elapsed_sec": round(elapsed, 1),
                "report": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
