#!/usr/bin/env python3
"""Aggregate strong HF math + coding datasets into a multi-reward slime mix.

Unified JSONL schema (one row = one RL sample):
  {
    "prompt": "...",
    "answer": "...",                 # math gold / empty for coding
    "unit_tests": {...} | null,      # coding only
    "reward_name": "math"|"unit_tests"|...,
    "domain": "math"|"coding",
    "source": "...",
    "dataset": "...",
    "hash_id": "...",
    "difficulty": float,             # lower = easier (curriculum)
    ...
  }

Use with slime config:
  reward: multi   # per-sample dispatch (math vs unit_tests)

Sources (default caps keep VRAM/time sane for 1×24GB GRPO):
  Coding (unit_tests):
    - local Nemotron competitive coding train (already converted)
    - open-r1/verifiable-coding-problems-python (stdin/stdout when available)
    - deepmind/code_contests (public/generated tests, Python-capable)
  Math (math reward / boxed / ####):
    - openai/gsm8k
    - DigitalLearningGmbH/MATH-lighteval (or qwedsacf/competition_math)
    - meta-math/MetaMathQA (sampled)
    - open-r1/OpenR1-Math-220k (sampled, verified answers when present)
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
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]

_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_HASH_RE = re.compile(r"####\s*([^\n]+)")
_SOURCE_RANK = {
    "aizu": 0,
    "atcoder": 1,
    "hackerearth": 2,
    "codechef": 3,
    "codeforces": 4,
    "apps": 3,
    "taco": 3,
    "code_contests": 3,
    "mbpp": 1,
    "humaneval": 1,
    "gsm8k": 0,
    "math": 2,
    "metamath": 1,
    "openr1_math": 2,
    "numina": 2,
}


def _hash_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:32]


def _clamp_prompt(text: str, max_chars: int) -> str | None:
    text = (text or "").strip()
    if len(text) < 32:
        return None
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n\n[truncated]"
    return text


def _extract_boxed(text: str) -> str | None:
    matches = _BOXED_RE.findall(text or "")
    return matches[-1].strip() if matches else None


def _extract_hash_answer(text: str) -> str | None:
    matches = _HASH_RE.findall(text or "")
    return matches[-1].strip() if matches else None


def _math_prompt(problem: str) -> str:
    return (
        "Solve the following problem carefully. "
        "Show brief reasoning, then put the final answer in \\boxed{...}.\n\n"
        f"{problem.strip()}"
    )


def _coding_prompt(problem: str) -> str:
    return (
        "Write a Python solution that reads from stdin and writes to stdout.\n"
        "Provide the final solution in a single ```python code block.\n\n"
        f"{problem.strip()}"
    )


def _difficulty_coding(source: str, n_tests: int, prompt_len: int) -> float:
    src = float(_SOURCE_RANK.get(str(source).lower(), 3))
    return src * 1_000_000.0 + float(n_tests) * 1_000.0 + float(prompt_len)


def _difficulty_math(source: str, level: Any, prompt_len: int) -> float:
    src = float(_SOURCE_RANK.get(str(source).lower(), 2))
    lvl = 3.0
    if level is not None:
        s = str(level)
        m = re.search(r"(\d+)", s)
        if m:
            lvl = float(m.group(1))
        elif isinstance(level, (int, float)):
            lvl = float(level)
    return src * 1_000_000.0 + lvl * 50_000.0 + float(prompt_len)


def _unit_tests_ok(ut: dict[str, Any] | None, *, min_tests: int = 1) -> bool:
    if not isinstance(ut, dict):
        return False
    inputs = ut.get("inputs") or []
    outputs = ut.get("outputs") or []
    if not inputs or not outputs or len(inputs) != len(outputs):
        return False
    return len(inputs) >= min_tests


def _trim_unit_tests(ut: dict[str, Any], max_tests: int) -> dict[str, list[str]]:
    inputs = [str(x) for x in (ut.get("inputs") or [])[:max_tests]]
    outputs = [str(x) for x in (ut.get("outputs") or [])[:max_tests]]
    return {"inputs": inputs, "outputs": outputs}


def _record(
    *,
    prompt: str,
    answer: str,
    reward_name: str,
    domain: str,
    source: str,
    dataset: str,
    difficulty: float,
    unit_tests: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    max_unit_tests: int = 8,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "prompt": prompt,
        "answer": answer,
        "reward_name": reward_name,
        "domain": domain,
        "source": source,
        "dataset": dataset,
        "difficulty": float(difficulty),
        "hash_id": _hash_id(domain, dataset, prompt[:2000], answer[:200]),
    }
    if unit_tests is not None:
        row["unit_tests"] = unit_tests
        row["max_unit_tests"] = max_unit_tests
        row["unit_test_timeout_sec"] = 2.0
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------


def load_local_nemotron(
    path: Path,
    *,
    max_samples: int | None,
    max_prompt_chars: int,
    max_tests: int,
) -> Iterator[dict[str, Any]]:
    if not path.exists():
        print(f"  skip missing {path}", file=sys.stderr)
        return
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if max_samples is not None and n >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            prompt = _clamp_prompt(str(raw.get("prompt") or ""), max_prompt_chars)
            ut = raw.get("unit_tests")
            if not prompt or not _unit_tests_ok(ut):
                continue
            ut = _trim_unit_tests(ut, max_tests)
            source = str(raw.get("source") or "nemotron")
            yield _record(
                prompt=prompt,
                answer=str(raw.get("answer") or ""),
                reward_name="unit_tests",
                domain="coding",
                source=source,
                dataset=str(raw.get("dataset") or "nemotron_coding"),
                difficulty=_difficulty_coding(source, len(ut["inputs"]), len(prompt)),
                unit_tests=ut,
                max_unit_tests=max_tests,
                extra={"hash_id": raw.get("hash_id") or _hash_id(prompt)},
            )
            n += 1


def load_openr1_verifiable_coding(
    *,
    max_samples: int | None,
    max_prompt_chars: int,
    max_tests: int,
) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    print("  loading open-r1/verifiable-coding-problems-python ...", file=sys.stderr)
    ds = load_dataset(
        "open-r1/verifiable-coding-problems-python",
        split="train",
        streaming=True,
    )
    n = 0
    for raw in ds:
        if max_samples is not None and n >= max_samples:
            break
        vi = raw.get("verification_info") or {}
        # Expect language + test cases in various shapes.
        language = str(vi.get("language") or "").lower()
        if language and language not in {"python", "python3", "py"}:
            continue
        tests = vi.get("test_cases") or vi.get("tests") or []
        inputs: list[str] = []
        outputs: list[str] = []
        if isinstance(tests, list):
            for t in tests:
                if not isinstance(t, dict):
                    continue
                # stdin/stdout style only (not assert-style fn tests).
                if "input" in t and ("output" in t or "expected" in t):
                    inputs.append(str(t["input"]))
                    outputs.append(str(t.get("output", t.get("expected", ""))))
        if not inputs:
            # some rows nest under fn_name style — skip (not stdin/stdout)
            continue
        problem = str(raw.get("problem_statement") or "").strip()
        prompt = _clamp_prompt(_coding_prompt(problem), max_prompt_chars)
        if not prompt:
            continue
        ut = _trim_unit_tests({"inputs": inputs, "outputs": outputs}, max_tests)
        if not _unit_tests_ok(ut):
            continue
        source = str(raw.get("source") or "openr1")
        yield _record(
            prompt=prompt,
            answer="",
            reward_name="unit_tests",
            domain="coding",
            source=source,
            dataset="open-r1/verifiable-coding-problems-python",
            difficulty=_difficulty_coding(source, len(ut["inputs"]), len(prompt)),
            unit_tests=ut,
            max_unit_tests=max_tests,
            extra={"problem_id": raw.get("problem_id")},
        )
        n += 1
        if n % 500 == 0:
            print(f"    openr1 coding kept={n}", file=sys.stderr)


def load_code_contests(
    *,
    max_samples: int | None,
    max_prompt_chars: int,
    max_tests: int,
) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    print("  loading deepmind/code_contests ...", file=sys.stderr)
    ds = load_dataset("deepmind/code_contests", split="train", streaming=True)
    n = 0
    for raw in ds:
        if max_samples is not None and n >= max_samples:
            break
        # Prefer problems that have at least one Python solution (optional filter).
        sols = raw.get("solutions") or {}
        languages = sols.get("language") or []
        # code_contests language enum: 1=PYTHON often; keep if any solution exists or ignore.
        has_py = False
        if languages:
            # Common mapping in CF: PYTHON=1, CPP=2, etc. Accept if 1 present or unknown.
            has_py = any(int(x) == 1 for x in languages if str(x).isdigit() or isinstance(x, int))
        # Keep even without py solutions if tests exist — RL generates code.
        public = raw.get("public_tests") or {}
        generated = raw.get("generated_tests") or {}
        inputs = list(public.get("input") or []) + list(generated.get("input") or [])
        outputs = list(public.get("output") or []) + list(generated.get("output") or [])
        if not inputs or len(inputs) != len(outputs):
            continue
        desc = str(raw.get("description") or "").strip()
        prompt = _clamp_prompt(_coding_prompt(desc), max_prompt_chars)
        if not prompt:
            continue
        ut = _trim_unit_tests({"inputs": inputs, "outputs": outputs}, max_tests)
        if not _unit_tests_ok(ut):
            continue
        source = "code_contests"
        yield _record(
            prompt=prompt,
            answer="",
            reward_name="unit_tests",
            domain="coding",
            source=source,
            dataset="deepmind/code_contests",
            difficulty=_difficulty_coding(source, len(ut["inputs"]), len(prompt)),
            unit_tests=ut,
            max_unit_tests=max_tests,
            extra={
                "name": raw.get("name"),
                "cf_index": raw.get("cf_index"),
                "has_python_solutions": has_py,
            },
        )
        n += 1
        if n % 200 == 0:
            print(f"    code_contests kept={n}", file=sys.stderr)


def load_gsm8k(
    *,
    max_samples: int | None,
    max_prompt_chars: int,
) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    print("  loading openai/gsm8k ...", file=sys.stderr)
    ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    n = 0
    for raw in ds:
        if max_samples is not None and n >= max_samples:
            break
        q = str(raw.get("question") or "").strip()
        a = str(raw.get("answer") or "").strip()
        final = _extract_hash_answer(a) or a
        prompt = _clamp_prompt(_math_prompt(q), max_prompt_chars)
        if not prompt or not final:
            continue
        yield _record(
            prompt=prompt,
            answer=final,
            reward_name="math",
            domain="math",
            source="gsm8k",
            dataset="openai/gsm8k",
            difficulty=_difficulty_math("gsm8k", 1, len(prompt)),
        )
        n += 1


def load_math_lighteval(
    *,
    max_samples: int | None,
    max_prompt_chars: int,
) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    print("  loading DigitalLearningGmbH/MATH-lighteval ...", file=sys.stderr)
    try:
        ds = load_dataset(
            "DigitalLearningGmbH/MATH-lighteval",
            split="train",
            streaming=True,
        )
    except Exception:
        print("  fallback qwedsacf/competition_math ...", file=sys.stderr)
        ds = load_dataset("qwedsacf/competition_math", split="train", streaming=True)
    n = 0
    for raw in ds:
        if max_samples is not None and n >= max_samples:
            break
        problem = str(raw.get("problem") or "").strip()
        solution = str(raw.get("solution") or "").strip()
        final = _extract_boxed(solution)
        if not final:
            # last-line heuristic
            lines = [ln.strip() for ln in solution.splitlines() if ln.strip()]
            final = lines[-1] if lines else ""
        prompt = _clamp_prompt(_math_prompt(problem), max_prompt_chars)
        if not prompt or not final:
            continue
        level = raw.get("level") or raw.get("type")
        yield _record(
            prompt=prompt,
            answer=final,
            reward_name="math",
            domain="math",
            source="math",
            dataset="MATH",
            difficulty=_difficulty_math("math", level, len(prompt)),
            extra={"level": level, "type": raw.get("type")},
        )
        n += 1
        if n % 1000 == 0:
            print(f"    MATH kept={n}", file=sys.stderr)


def load_metamath(
    *,
    max_samples: int | None,
    max_prompt_chars: int,
) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    print("  loading meta-math/MetaMathQA ...", file=sys.stderr)
    ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    n = 0
    for raw in ds:
        if max_samples is not None and n >= max_samples:
            break
        q = str(raw.get("query") or raw.get("original_question") or "").strip()
        resp = str(raw.get("response") or "").strip()
        final = _extract_boxed(resp) or _extract_hash_answer(resp)
        if not final:
            # MetaMath often ends with "The answer is: X"
            m = re.search(r"the answer is[:\s]+(.+?)\.?$", resp, flags=re.I | re.S)
            final = m.group(1).strip() if m else ""
        prompt = _clamp_prompt(_math_prompt(q), max_prompt_chars)
        if not prompt or not final or len(final) > 200:
            continue
        yield _record(
            prompt=prompt,
            answer=final,
            reward_name="math",
            domain="math",
            source="metamath",
            dataset="meta-math/MetaMathQA",
            difficulty=_difficulty_math("metamath", 2, len(prompt)),
            extra={"type": raw.get("type")},
        )
        n += 1
        if n % 2000 == 0:
            print(f"    MetaMath kept={n}", file=sys.stderr)


def load_openr1_math(
    *,
    max_samples: int | None,
    max_prompt_chars: int,
) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    print("  loading open-r1/OpenR1-Math-220k ...", file=sys.stderr)
    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train", streaming=True)
    n = 0
    for raw in ds:
        if max_samples is not None and n >= max_samples:
            break
        # Prefer rows with math verification when present
        correctness = raw.get("correctness_math_verify")
        if isinstance(correctness, list) and correctness and not any(correctness):
            continue
        problem = str(raw.get("problem") or "").strip()
        answer = str(raw.get("answer") or "").strip()
        if not answer:
            answer = _extract_boxed(str(raw.get("solution") or "")) or ""
        prompt = _clamp_prompt(_math_prompt(problem), max_prompt_chars)
        if not prompt or not answer or len(answer) > 300:
            continue
        yield _record(
            prompt=prompt,
            answer=answer,
            reward_name="math",
            domain="math",
            source=str(raw.get("source") or "openr1_math"),
            dataset="open-r1/OpenR1-Math-220k",
            difficulty=_difficulty_math("openr1_math", raw.get("problem_type"), len(prompt)),
            extra={"problem_type": raw.get("problem_type")},
        )
        n += 1
        if n % 1000 == 0:
            print(f"    OpenR1-Math kept={n}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Mix / split / write
# ---------------------------------------------------------------------------


def _dedupe(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("hash_id") or _hash_id(row.get("prompt", "")))
        # Also collapse near-dup prompts
        pkey = _hash_id((row.get("prompt") or "")[:1500])
        if key in seen or pkey in seen:
            continue
        seen.add(key)
        seen.add(pkey)
        out.append(row)
    return out


def _balanced_sample(
    rows: list[dict[str, Any]],
    *,
    max_total: int,
    math_ratio: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    math_rows = [r for r in rows if r.get("domain") == "math"]
    code_rows = [r for r in rows if r.get("domain") == "coding"]
    rng.shuffle(math_rows)
    rng.shuffle(code_rows)

    n_math = min(len(math_rows), int(max_total * math_ratio))
    n_code = min(len(code_rows), max_total - n_math)
    # fill remainder from whichever has leftover
    picked = math_rows[:n_math] + code_rows[:n_code]
    rem = max_total - len(picked)
    if rem > 0:
        leftover = math_rows[n_math:] + code_rows[n_code:]
        rng.shuffle(leftover)
        picked.extend(leftover[:rem])
    rng.shuffle(picked)
    return picked


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_domain = Counter(r.get("domain") for r in rows)
    by_dataset = Counter(r.get("dataset") for r in rows)
    by_reward = Counter(r.get("reward_name") for r in rows)
    diffs = sorted(float(r.get("difficulty") or 0) for r in rows)
    return {
        "n": len(rows),
        "by_domain": dict(by_domain),
        "by_reward": dict(by_reward),
        "by_dataset": dict(by_dataset.most_common(30)),
        "difficulty_p10": diffs[int(0.1 * len(diffs))] if diffs else None,
        "difficulty_p50": diffs[len(diffs) // 2] if diffs else None,
        "difficulty_p90": diffs[int(0.9 * len(diffs))] if diffs else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "slime_mix",
    )
    parser.add_argument(
        "--nemotron-train",
        type=Path,
        default=ROOT / "data" / "nemotron_coding_train.jsonl",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-total", type=int, default=24000, help="Final train size cap")
    parser.add_argument("--bench-size", type=int, default=128)
    parser.add_argument("--math-ratio", type=float, default=0.45)
    parser.add_argument("--max-prompt-chars", type=int, default=6000)
    parser.add_argument("--max-tests", type=int, default=8)
    # Per-source caps (before balance)
    parser.add_argument("--cap-nemotron", type=int, default=10000)
    parser.add_argument("--cap-openr1-code", type=int, default=4000)
    parser.add_argument("--cap-code-contests", type=int, default=4000)
    parser.add_argument("--cap-gsm8k", type=int, default=7473)
    parser.add_argument("--cap-math", type=int, default=7500)
    parser.add_argument("--cap-metamath", type=int, default=8000)
    parser.add_argument("--cap-openr1-math", type=int, default=6000)
    parser.add_argument("--skip-code-contests", action="store_true")
    parser.add_argument("--skip-openr1-code", action="store_true")
    parser.add_argument("--skip-metamath", action="store_true")
    parser.add_argument("--skip-openr1-math", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []

    print("=== Coding sources ===", file=sys.stderr)
    rows.extend(
        load_local_nemotron(
            args.nemotron_train,
            max_samples=args.cap_nemotron,
            max_prompt_chars=args.max_prompt_chars,
            max_tests=args.max_tests,
        )
    )
    print(f"  after nemotron: {len(rows)}", file=sys.stderr)

    if not args.skip_openr1_code:
        try:
            rows.extend(
                load_openr1_verifiable_coding(
                    max_samples=args.cap_openr1_code,
                    max_prompt_chars=args.max_prompt_chars,
                    max_tests=args.max_tests,
                )
            )
        except Exception as exc:
            print(f"  openr1 coding failed: {exc}", file=sys.stderr)
        print(f"  after openr1 code: {len(rows)}", file=sys.stderr)

    if not args.skip_code_contests:
        try:
            rows.extend(
                load_code_contests(
                    max_samples=args.cap_code_contests,
                    max_prompt_chars=args.max_prompt_chars,
                    max_tests=args.max_tests,
                )
            )
        except Exception as exc:
            print(f"  code_contests failed: {exc}", file=sys.stderr)
        print(f"  after code_contests: {len(rows)}", file=sys.stderr)

    print("=== Math sources ===", file=sys.stderr)
    try:
        rows.extend(
            load_gsm8k(
                max_samples=args.cap_gsm8k,
                max_prompt_chars=args.max_prompt_chars,
            )
        )
    except Exception as exc:
        print(f"  gsm8k failed: {exc}", file=sys.stderr)
    print(f"  after gsm8k: {len(rows)}", file=sys.stderr)

    try:
        rows.extend(
            load_math_lighteval(
                max_samples=args.cap_math,
                max_prompt_chars=args.max_prompt_chars,
            )
        )
    except Exception as exc:
        print(f"  MATH failed: {exc}", file=sys.stderr)
    print(f"  after MATH: {len(rows)}", file=sys.stderr)

    if not args.skip_metamath:
        try:
            rows.extend(
                load_metamath(
                    max_samples=args.cap_metamath,
                    max_prompt_chars=args.max_prompt_chars,
                )
            )
        except Exception as exc:
            print(f"  MetaMath failed: {exc}", file=sys.stderr)
        print(f"  after MetaMath: {len(rows)}", file=sys.stderr)

    if not args.skip_openr1_math:
        try:
            rows.extend(
                load_openr1_math(
                    max_samples=args.cap_openr1_math,
                    max_prompt_chars=args.max_prompt_chars,
                )
            )
        except Exception as exc:
            print(f"  OpenR1-Math failed: {exc}", file=sys.stderr)
        print(f"  after OpenR1-Math: {len(rows)}", file=sys.stderr)

    print(f"raw total={len(rows)}; deduping...", file=sys.stderr)
    rows = _dedupe(rows)
    print(f"deduped={len(rows)}", file=sys.stderr)

    # Hold out stratified bench first (never in train).
    math_rows = [r for r in rows if r.get("domain") == "math"]
    code_rows = [r for r in rows if r.get("domain") == "coding"]
    rng.shuffle(math_rows)
    rng.shuffle(code_rows)
    n_bench_math = min(len(math_rows), args.bench_size // 2)
    n_bench_code = min(len(code_rows), args.bench_size - n_bench_math)
    bench = math_rows[:n_bench_math] + code_rows[:n_bench_code]
    rng.shuffle(bench)
    train_pool = math_rows[n_bench_math:] + code_rows[n_bench_code:]

    train = _balanced_sample(
        train_pool,
        max_total=args.max_total,
        math_ratio=args.math_ratio,
        rng=rng,
    )

    # Curriculum-sorted copies (easy→hard) for progressive rounds.
    train_sorted = sorted(train, key=lambda r: float(r.get("difficulty") or 0.0))
    n = len(train_sorted)
    easy = train_sorted[: max(1, int(n * 0.40))]
    medium = train_sorted[int(n * 0.15) : max(int(n * 0.15) + 1, int(n * 0.70))]
    hard = train_sorted[int(n * 0.35) :]

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "train.jsonl", train)
    _write_jsonl(out / "train_easy.jsonl", easy)
    _write_jsonl(out / "train_medium.jsonl", medium)
    _write_jsonl(out / "train_hard.jsonl", hard)
    _write_jsonl(out / "bench.jsonl", bench)

    meta = {
        "train": _stats(train),
        "bench": _stats(bench),
        "easy": _stats(easy),
        "medium": _stats(medium),
        "hard": _stats(hard),
        "config": {
            "max_total": args.max_total,
            "math_ratio": args.math_ratio,
            "seed": args.seed,
            "reward": "multi",
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote mix under {out}", file=sys.stderr)
    return 0 if train and bench else 1


if __name__ == "__main__":
    raise SystemExit(main())
