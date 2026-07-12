#!/usr/bin/env python3
"""Build an elite multi-domain coding mix for slime multi-reward RL.

Domains (all verifiable):
  1. contest     — stdin/stdout unit_tests (Nemotron / code_contests / open-r1)
  2. codebase    — multi-file pytest packages (synthetic)
  3. function    — MBPP / MBPP+ / HumanEval-style assert_tests
  4. library     — BigCodeBench-hard style (assert/check when runnable)

Output under data/elite_coding/:
  train.jsonl, bench.jsonl, train_{easy,medium,hard,mixed}.jsonl, warmup_multi_easy.jsonl, meta.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:28]


def _write(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def _diff(r: dict[str, Any]) -> float:
    if r.get("difficulty") is not None:
        try:
            return float(r["difficulty"])
        except (TypeError, ValueError):
            pass
    domain = str(r.get("domain") or "")
    # easy → hard-ish defaults for curriculum
    base = {
        "function": 0.5e6,
        "library": 1.0e6,
        "codebase": 1.5e6,
        "coding": 2.5e6,
        "contest": 2.5e6,
    }.get(domain, 2.0e6)
    return base + len(str(r.get("prompt") or "")) * 0.01


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_existing_contest(max_n: int) -> list[dict[str, Any]]:
    paths = [
        ROOT / "data/slime_coding_opt/train.jsonl",
        ROOT / "data/slime_coding_complex/train.jsonl",
        ROOT / "data/nemotron_coding_train.jsonl",
    ]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for p in paths:
        for r in load_jsonl(p):
            if r.get("reward_name") == "codebase_tests":
                continue
            ut = r.get("unit_tests") or {}
            if not (ut.get("inputs") and ut.get("outputs")):
                continue
            h = str(r.get("hash_id") or _hash(r.get("prompt", "")[:800]))
            if h in seen:
                continue
            seen.add(h)
            row = dict(r)
            row["reward_name"] = "unit_tests"
            row["domain"] = "contest"
            row.setdefault("dataset", str(p.name))
            row["difficulty"] = _diff(row)
            row["hash_id"] = h
            out.append(row)
            if len(out) >= max_n:
                return out
    return out


def load_existing_codebases(max_n: int) -> list[dict[str, Any]]:
    paths = [
        ROOT / "data/synthetic_codebases/train.jsonl",
        ROOT / "data/slime_coding_complex/train.jsonl",
    ]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for p in paths:
        for r in load_jsonl(p):
            if r.get("reward_name") not in {None, "codebase_tests"} and r.get("domain") != "codebase":
                if not r.get("codebase"):
                    continue
            if not r.get("codebase"):
                continue
            h = str(r.get("hash_id") or _hash(json.dumps(r.get("codebase"), sort_keys=True)[:500]))
            if h in seen:
                continue
            seen.add(h)
            row = dict(r)
            row["reward_name"] = "codebase_tests"
            row["domain"] = "codebase"
            row["difficulty"] = _diff(row)
            row["hash_id"] = h
            out.append(row)
            if len(out) >= max_n:
                return out
    return out


def load_mbpp(max_n: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    print("  loading MBPP...", file=sys.stderr)
    out: list[dict[str, Any]] = []
    for split in ("train", "test", "validation"):
        try:
            ds = load_dataset(
                "google-research-datasets/mbpp", "full", split=split, streaming=True
            )
        except Exception:
            continue
        for raw in ds:
            text = str(raw.get("text") or "").strip()
            tests = list(raw.get("test_list") or [])
            if not text or not tests:
                continue
            setup = str(raw.get("test_setup_code") or "")
            challenge = list(raw.get("challenge_test_list") or [])
            all_tests = tests + challenge
            prompt = (
                "Write a Python function that solves the following problem. "
                "Put the full solution in a ```python code block.\n\n"
                f"{text}\n"
            )
            row = {
                "prompt": prompt,
                "answer": str(raw.get("code") or ""),
                "reward_name": "assert_tests",
                "domain": "function",
                "source": "mbpp",
                "dataset": "google-research-datasets/mbpp",
                "assert_tests": [str(t) for t in all_tests],
                "test_setup": setup,
                "task_id": raw.get("task_id"),
                "test_timeout_sec": 3.0,
                "difficulty": 0.4e6 + len(all_tests) * 1e3,
                "hash_id": _hash("mbpp", str(raw.get("task_id")), text[:200]),
            }
            out.append(row)
            if len(out) >= max_n:
                return out
    return out


def load_mbppplus(max_n: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    print("  loading MBPP+...", file=sys.stderr)
    out: list[dict[str, Any]] = []
    try:
        ds = load_dataset("evalplus/mbppplus", split="test", streaming=True)
    except Exception as exc:
        print(f"  mbppplus failed: {exc}", file=sys.stderr)
        return out
    for raw in ds:
        prompt = str(raw.get("prompt") or "").strip()
        tests = list(raw.get("test_list") or [])
        if not prompt or not tests:
            continue
        # MBPP+ also has a combined test string; prefer asserts
        row = {
            "prompt": (
                "Write a Python function solving this. "
                "Return a complete ```python code block.\n\n" + prompt
            ),
            "answer": str(raw.get("code") or ""),
            "reward_name": "assert_tests",
            "domain": "function",
            "source": "mbppplus",
            "dataset": "evalplus/mbppplus",
            "assert_tests": [str(t) for t in tests],
            "test_setup": "\n".join(str(x) for x in (raw.get("test_imports") or [])),
            "task_id": raw.get("task_id"),
            "test_timeout_sec": 4.0,
            "difficulty": 0.7e6 + len(tests) * 500,
            "hash_id": _hash("mbppplus", str(raw.get("task_id")), prompt[:200]),
        }
        out.append(row)
        if len(out) >= max_n:
            break
    return out


def load_humaneval(max_n: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    print("  loading HumanEval...", file=sys.stderr)
    out: list[dict[str, Any]] = []
    try:
        ds = load_dataset("openai/openai_humaneval", split="test", streaming=True)
    except Exception as exc:
        print(f"  humaneval failed: {exc}", file=sys.stderr)
        return out
    for raw in ds:
        prefix = str(raw.get("prompt") or "")
        entry = str(raw.get("entry_point") or "")
        test = str(raw.get("test") or "")
        if not prefix or not entry or not test:
            continue
        prompt = (
            "Complete the following Python function. "
            "Output the full function in a ```python code block "
            "(include the function signature).\n\n"
            f"{prefix}"
        )
        row = {
            "prompt": prompt,
            "answer": prefix + str(raw.get("canonical_solution") or ""),
            "reward_name": "assert_tests",
            "domain": "function",
            "source": "humaneval",
            "dataset": "openai/openai_humaneval",
            "code_prefix": "",
            "entry_point": entry,
            "check_code": test,
            "task_id": raw.get("task_id"),
            "test_timeout_sec": 4.0,
            "difficulty": 0.9e6,
            "hash_id": _hash("humaneval", str(raw.get("task_id"))),
        }
        out.append(row)
        if len(out) >= max_n:
            break
    return out


def load_bigcodebench_hard(max_n: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    print("  loading BigCodeBench-Hard...", file=sys.stderr)
    out: list[dict[str, Any]] = []
    try:
        ds = load_dataset("bigcode/bigcodebench-hard", split="v0.1.0_hf", streaming=True)
    except Exception as exc:
        print(f"  bigcodebench failed: {exc}", file=sys.stderr)
        return out
    for raw in ds:
        instruct = str(raw.get("instruct_prompt") or raw.get("complete_prompt") or "").strip()
        test = str(raw.get("test") or "")
        entry = str(raw.get("entry_point") or "")
        code_prompt = str(raw.get("code_prompt") or "")
        if not instruct or not test:
            continue
        # Many BCB tests are unittest; wrap as check if entry exists
        prompt = (
            "Implement the required Python solution for this practical coding task. "
            "Provide a complete ```python module (imports + functions/classes).\n\n"
            f"{instruct}\n"
        )
        if code_prompt:
            prompt += f"\nStarter / signature hints:\n```python\n{code_prompt}\n```\n"
        row = {
            "prompt": prompt[:6000],
            "answer": str(raw.get("canonical_solution") or ""),
            "reward_name": "assert_tests",
            "domain": "library",
            "source": "bigcodebench-hard",
            "dataset": "bigcode/bigcodebench-hard",
            "entry_point": entry or "task_func",
            "check_code": test,
            "code_prefix": code_prompt,
            "task_id": raw.get("task_id"),
            "test_timeout_sec": 6.0,
            "difficulty": 1.8e6 + len(instruct) * 0.05,
            "hash_id": _hash("bcb", str(raw.get("task_id"))),
        }
        out.append(row)
        if len(out) >= max_n:
            break
    return out


def ensure_more_codebases(target: int) -> list[dict[str, Any]]:
    """Generate extra synthetic packages if below target."""
    existing = load_existing_codebases(target)
    if len(existing) >= target:
        return existing[:target]
    print(f"  generating extra synthetic codebases (have {len(existing)}, want {target})...", file=sys.stderr)
    import importlib.util

    gen_path = ROOT / "scripts" / "generate_synthetic_codebases.py"
    spec = importlib.util.spec_from_file_location("generate_synthetic_codebases", gen_path)
    if spec is None or spec.loader is None:
        print("  could not load generator module", file=sys.stderr)
        return existing[:target]
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    generators = mod.GENERATORS

    need = target - len(existing)
    seen = {r.get("hash_id") for r in existing}
    rng = random.Random(99)
    i = 0
    while need > 0 and i < need * 3:
        gen = generators[i % len(generators)]
        t = gen(random.Random(rng.randint(0, 10**9)), 10_000 + i)
        if t["hash_id"] in seen:
            i += 1
            continue
        seen.add(t["hash_id"])
        existing.append(t)
        need -= 1
        i += 1
        if i % 200 == 0:
            print(f"    extra gen {i}, total codebase {len(existing)}", file=sys.stderr)
    return existing[:target]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "elite_coding")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cap-contest", type=int, default=6000)
    parser.add_argument("--cap-codebase", type=int, default=6000)
    parser.add_argument("--cap-mbpp", type=int, default=2000)
    parser.add_argument("--cap-mbppplus", type=int, default=500)
    parser.add_argument("--cap-humaneval", type=int, default=200)
    parser.add_argument("--cap-bcb", type=int, default=400)
    parser.add_argument("--max-train", type=int, default=14000)
    parser.add_argument("--bench-size", type=int, default=200)
    parser.add_argument("--math-warmup", type=int, default=2500)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pools: dict[str, list[dict[str, Any]]] = {}

    print("=== Contest (unit_tests) ===", file=sys.stderr)
    pools["contest"] = load_existing_contest(args.cap_contest)
    print(f"  n={len(pools['contest'])}", file=sys.stderr)

    print("=== Multi-file codebases ===", file=sys.stderr)
    pools["codebase"] = ensure_more_codebases(args.cap_codebase)
    print(f"  n={len(pools['codebase'])}", file=sys.stderr)

    print("=== Function / library (assert_tests) ===", file=sys.stderr)
    pools["mbpp"] = load_mbpp(args.cap_mbpp)
    print(f"  mbpp n={len(pools['mbpp'])}", file=sys.stderr)
    pools["mbppplus"] = load_mbppplus(args.cap_mbppplus)
    print(f"  mbpp+ n={len(pools['mbppplus'])}", file=sys.stderr)
    pools["humaneval"] = load_humaneval(args.cap_humaneval)
    print(f"  humaneval n={len(pools['humaneval'])}", file=sys.stderr)
    pools["bcb"] = load_bigcodebench_hard(args.cap_bcb)
    print(f"  bcb n={len(pools['bcb'])}", file=sys.stderr)

    # Balanced sample into train pool
    domain_rows: dict[str, list[dict[str, Any]]] = {
        "contest": pools["contest"],
        "codebase": pools["codebase"],
        "function": pools["mbpp"] + pools["mbppplus"] + pools["humaneval"],
        "library": pools["bcb"],
    }
    for k, v in domain_rows.items():
        rng.shuffle(v)
        print(f"domain {k}: {len(v)}", file=sys.stderr)

    # Target domain mix for elite breadth
    # contest 35%, codebase 35%, function 20%, library 10%
    targets = {
        "contest": int(args.max_train * 0.35),
        "codebase": int(args.max_train * 0.35),
        "function": int(args.max_train * 0.20),
        "library": int(args.max_train * 0.10),
    }
    train: list[dict[str, Any]] = []
    for dom, n in targets.items():
        take = domain_rows[dom][: min(n, len(domain_rows[dom]))]
        train.extend(take)
    # fill remainder from largest leftover
    rem = args.max_train - len(train)
    if rem > 0:
        leftover: list[dict[str, Any]] = []
        for dom, rows in domain_rows.items():
            leftover.extend(rows[targets[dom] :])
        rng.shuffle(leftover)
        train.extend(leftover[:rem])
    rng.shuffle(train)

    # Stratified bench: hold out per domain
    bench: list[dict[str, Any]] = []
    train_ids = {r["hash_id"] for r in train}
    # rebuild bench from unused + slice from each domain
    per = max(1, args.bench_size // 4)
    used_for_bench: set[str] = set()
    for dom, rows in domain_rows.items():
        # prefer rows not over-represented; take from end of shuffled
        picked = 0
        for r in reversed(rows):
            h = r["hash_id"]
            if h in used_for_bench:
                continue
            bench.append(r)
            used_for_bench.add(h)
            picked += 1
            if picked >= per:
                break
    # remove bench from train
    train = [r for r in train if r["hash_id"] not in used_for_bench]
    rng.shuffle(bench)
    bench = bench[: args.bench_size]

    # Curriculum
    ranked = sorted(train, key=_diff)
    n = len(ranked)
    easy = ranked[: max(1, int(n * 0.35))]
    medium = ranked[int(n * 0.12) : max(int(n * 0.12) + 1, int(n * 0.70))]
    hard = ranked[int(n * 0.40) :]
    mixed = ranked

    # Warmup: functions (dense) + easy codebases + a bit of contest
    warmup = (
        domain_rows["function"][:800]
        + domain_rows["codebase"][:800]
        + domain_rows["contest"][:600]
        + load_jsonl(ROOT / "data/slime_coding_opt/warmup_multi_easy.jsonl", args.math_warmup)
    )
    rng.shuffle(warmup)
    warmup = warmup[: max(2000, args.math_warmup)]

    out = args.out_dir
    stats = {
        "train": _write(out / "train.jsonl", train),
        "bench": _write(out / "bench.jsonl", bench),
        "easy": _write(out / "train_easy.jsonl", easy),
        "medium": _write(out / "train_medium.jsonl", medium),
        "hard": _write(out / "train_hard.jsonl", hard),
        "mixed": _write(out / "train_mixed.jsonl", mixed),
        "warmup": _write(out / "warmup_multi_easy.jsonl", warmup),
        "by_domain_train": dict(Counter(r.get("domain") for r in train)),
        "by_reward_train": dict(Counter(r.get("reward_name") for r in train)),
        "by_dataset_train": dict(Counter(r.get("dataset") for r in train).most_common(20)),
        "by_domain_bench": dict(Counter(r.get("domain") for r in bench)),
        "by_reward_bench": dict(Counter(r.get("reward_name") for r in bench)),
        "pools": {k: len(v) for k, v in pools.items()},
    }
    (out / "meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Wrote elite mix → {out}", file=sys.stderr)
    return 0 if train and bench else 1


if __name__ == "__main__":
    raise SystemExit(main())
