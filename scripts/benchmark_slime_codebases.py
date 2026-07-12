#!/usr/bin/env python3
"""Benchmark a model/adapter on thousands of complex multi-file codebase tasks.

Judges with pytest over synthetic packages (reward_name=codebase_tests).
Supports --limit for quick checks and large full-suite runs.
"""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="/home/c/models/Qwen3-8B")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/synthetic_codebases/bench.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=None, help="None = all rows")
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/bench/codebase_bench_report.json"),
    )
    parser.add_argument(
        "--gold-only",
        action="store_true",
        help="Only verify gold solutions (no model) — dataset integrity",
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    args = parser.parse_args()

    from seiso.slime_single_gpu.codebase_judge import (
        codebase_tests_reward,
        extract_file_map,
        gold_passes,
    )
    from seiso.slime_single_gpu.rewards import multi_reward

    rows = _load_jsonl(args.dataset, args.limit)
    if not rows:
        print(f"no rows in {args.dataset}", file=sys.stderr)
        return 1
    print(f"Loaded {len(rows)} codebase tasks from {args.dataset}", flush=True)

    if args.gold_only:
        ok = fail = 0
        t0 = time.time()
        for i, sample in enumerate(rows):
            good = gold_passes(sample)
            ok += int(good)
            fail += int(not good)
            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(rows)}] gold ok={ok} fail={fail}", flush=True)
        report = {
            "mode": "gold_only",
            "n": len(rows),
            "gold_ok": ok,
            "gold_fail": fail,
            "mean_pass_rate": ok / len(rows),
            "elapsed_sec": time.time() - t0,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if fail == 0 else 2

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"Loading model {args.model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=args.trust_remote_code, use_fast=True
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

        print(f"Loading adapter {args.adapter}", flush=True)
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    scores: list[float] = []
    results: list[dict[str, Any]] = []
    t0 = time.time()
    for i, sample in enumerate(rows):
        prompt = str(sample["prompt"])
        if getattr(tokenizer, "chat_template", None):
            try:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            text = prompt
        encoded = tokenizer(
            text,
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
        score = float(
            max(
                multi_reward(completion, sample),
                codebase_tests_reward(completion, sample),
            )
        )
        files = extract_file_map(
            completion,
            list((sample.get("codebase") or {}).get("target_files") or []),
        )
        scores.append(score)
        results.append(
            {
                "index": i,
                "hash_id": sample.get("hash_id"),
                "family": sample.get("family"),
                "pass_rate": score,
                "n_files_emitted": len(files),
                "files": list(files.keys()),
                "completion_chars": len(completion),
                "completion_preview": completion[:300],
            }
        )
        print(
            f"[{i+1}/{len(rows)}] pass={score:.2f} family={sample.get('family')} "
            f"files={list(files.keys())}",
            flush=True,
        )
        del out, encoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean = sum(scores) / len(scores)
    perfect = sum(1 for s in scores if s >= 0.999) / len(scores)
    any_pass = sum(1 for s in scores if s > 0.05) / len(scores)
    report = {
        "model_id": args.model_id,
        "adapter": str(args.adapter) if args.adapter else None,
        "dataset": str(args.dataset),
        "mode": "codebase_tests",
        "n": len(rows),
        "mean_pass_rate": mean,
        "perfect_rate": perfect,
        "any_pass_rate": any_pass,
        # Compatibility with coding loop quality gates
        "code_extract_rate": sum(1 for r in results if r["n_files_emitted"] > 0) / len(results),
        "coding_mean": mean,
        "elapsed_sec": time.time() - t0,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
