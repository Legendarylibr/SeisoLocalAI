"""High-level verifiable RL data generation for slime / distill-RL.

Hardcoded 30-line smoke JSONL does not produce meaningful GRPO signal:
models either pass everything or fail uniformly, dynamic sampling drops all
groups, and training aborts with ``no_trainable_groups``.

This module builds **large, deterministic, checkable** prompt corpora:

* **numeric** — arithmetic + multi-step word problems with exact answers
* **choice** — multiple-choice facts/logic with letter labels
* **code** — unit-test-grounded programs (via :mod:`seiso.rl_verify.code_corpus`)

Each row is slime-ready: ``prompt``, ``answer`` / ``tests``, ``benchmark``,
and difficulty tags. Same ``seed`` ⇒ same corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config / mix
# ---------------------------------------------------------------------------

_DEFAULT_STREAM_MIX = {"numeric": 0.50, "choice": 0.20, "code": 0.30}
_DEFAULT_DIFFICULTY_MIX = {"easy": 0.35, "medium": 0.45, "hard": 0.20}
_STREAMS = frozenset({"numeric", "choice", "code"})
_DIFFICULTIES = frozenset({"easy", "medium", "hard"})


@dataclass(frozen=True)
class DataGenConfig:
    """High-level RL corpus generation request."""

    count: int = 500
    seed: int = 0
    # Stream mix: numeric / choice / code (weights normalized).
    mix: str | dict[str, float] = "numeric:0.5,choice:0.2,code:0.3"
    # Difficulty within each stream.
    difficulty: str | dict[str, float] = "easy:0.35,medium:0.45,hard:0.20"
    # Require thinking-style instruction in the prompt (matches slime default).
    require_thinking_trace: bool = True
    thinking_instruction: str = (
        "Show your reasoning in <think>...</think>, then give the final answer."
    )
    # When code stream is non-zero, verify goldens via sandbox (slower, safer).
    verify_code: bool = True
    include_code_hand_catalog: bool = False


@dataclass(frozen=True)
class DataGenResult:
    """Generated rows plus light diagnostics for GRPO readiness."""

    rows: list[dict[str, Any]]
    stream_counts: dict[str, int]
    difficulty_counts: dict[str, int]
    seed: int

    @property
    def count(self) -> int:
        return len(self.rows)

    def summary(self) -> dict[str, Any]:
        answers = [
            str(r.get("label", r.get("answer", "")))
            for r in self.rows
            if (r.get("metadata") or {}).get("rm_type", r.get("reward")) != "code"
        ]
        unique_answers = len(set(answers))
        return {
            "count": self.count,
            "seed": self.seed,
            "stream_counts": dict(self.stream_counts),
            "difficulty_counts": dict(self.difficulty_counts),
            "unique_noncode_answers": unique_answers,
            "answer_diversity": (
                unique_answers / max(1, len(answers)) if answers else 0.0
            ),
        }


def parse_weight_mix(
    spec: str | dict[str, float] | None,
    *,
    allowed: frozenset[str],
    default: dict[str, float],
) -> dict[str, float]:
    """Parse ``a:0.5,b:0.5`` or a dict into a normalized positive mix."""
    if spec is None:
        raw = dict(default)
    elif isinstance(spec, dict):
        raw = {str(k).lower().strip(): float(v) for k, v in spec.items()}
    else:
        raw = {}
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(f"bad mix fragment {part!r}; expected key:weight")
            key, weight = part.split(":", 1)
            raw[key.strip().lower()] = float(weight)
    unknown = set(raw) - set(allowed)
    if unknown:
        raise ValueError(
            f"unknown mix keys {sorted(unknown)}; allowed: {sorted(allowed)}"
        )
    if not raw:
        raw = dict(default)
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("mix weights must sum to > 0")
    return {k: v / total for k, v in raw.items() if v > 0}


def _stable_rng(*parts: str) -> random.Random:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


# ---------------------------------------------------------------------------
# Numeric stream
# ---------------------------------------------------------------------------


def _thinking_suffix(cfg: DataGenConfig) -> str:
    if not cfg.require_thinking_trace:
        return ""
    return f"\n\n{cfg.thinking_instruction}"


def to_slime_prompt_row(
    content: str,
    label: str,
    *,
    rm_type: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a THUDM/slime-compatible JSONL row.

    Slime loads ``prompt`` (optionally chat messages) + ``label`` and optional
    ``metadata``. Seiso also keeps ``answer`` as an alias of ``label`` so local
    trainers can use either field name.
    """
    meta = dict(metadata or {})
    meta.setdefault("rm_type", rm_type)
    meta.setdefault("source_name", "seiso.data_gen")
    row: dict[str, Any] = {
        # slime-native chat form (use with apply_chat_template)
        "prompt": [{"role": "user", "content": content}],
        "label": label,
        "answer": label,  # Seiso alias
        "metadata": meta,
    }
    # Top-level checker hints for Seiso verifier (also mirrored in metadata).
    row["reward"] = rm_type
    row["benchmark"] = meta.get("benchmark", rm_type)
    if "tests" in meta:
        row["tests"] = meta["tests"]
    if "solution" in meta:
        row["solution"] = meta["solution"]
    if "timeout_s" in meta:
        row["timeout_s"] = meta["timeout_s"]
    return row


def _numeric_easy(rng: random.Random, index: int) -> tuple[str, str]:
    op = rng.choice(["+", "-", "*", "//"])
    if op == "+":
        a, b = rng.randint(1, 40), rng.randint(1, 40)
        return f"What is {a} + {b}?", str(a + b)
    if op == "-":
        a, b = rng.randint(10, 60), rng.randint(1, 20)
        if b > a:
            a, b = b, a
        return f"What is {a} - {b}?", str(a - b)
    if op == "*":
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        return f"What is {a} times {b}?", str(a * b)
    # integer division that divides evenly
    b = rng.randint(2, 12)
    q = rng.randint(2, 15)
    a = b * q
    return f"What is {a} divided by {b}? (integer result)", str(q)


def _numeric_medium(rng: random.Random, index: int) -> tuple[str, str]:
    kind = rng.randint(0, 4)
    if kind == 0:
        a, b, c = rng.randint(2, 20), rng.randint(2, 15), rng.randint(1, 10)
        ans = a * b + c
        return (
            f"Compute {a} × {b} + {c}.",
            str(ans),
        )
    if kind == 1:
        n = rng.randint(5, 25)
        take = rng.randint(1, n - 1)
        remain = n - take
        who = rng.choice(["Sam", "Ava", "Lee", "Kai", "Mia"])
        item = rng.choice(["apples", "pencils", "coins", "books", "stickers"])
        return (
            f"A box has {n} {item}. {who} removes {take}. How many remain?",
            str(remain),
        )
    if kind == 2:
        price = rng.randint(3, 20)
        qty = rng.randint(2, 8)
        pay = price * qty + rng.randint(1, 15)
        change = pay - price * qty
        return (
            f"Each ticket costs ${price}. You buy {qty} and pay ${pay}. "
            f"How many dollars in change?",
            str(change),
        )
    if kind == 3:
        a, b = rng.randint(10, 40), rng.randint(10, 40)
        ans = abs(a - b)
        return (
            f"What is the absolute difference between {a} and {b}?",
            str(ans),
        )
    # percent of integer that stays integer
    pct = rng.choice([10, 20, 25, 50])
    base = rng.randint(2, 20) * (100 // math.gcd(pct, 100))
    ans = base * pct // 100
    return (f"What is {pct}% of {base}?", str(ans))


def _numeric_hard(rng: random.Random, index: int) -> tuple[str, str]:
    kind = rng.randint(0, 3)
    if kind == 0:
        # multi-step word problem
        start = rng.randint(20, 80)
        add = rng.randint(5, 25)
        sub = rng.randint(3, 20)
        mult = rng.randint(2, 4)
        ans = (start + add - sub) * mult
        return (
            f"Start with {start}. Add {add}, subtract {sub}, then multiply by {mult}. "
            f"What is the result?",
            str(ans),
        )
    if kind == 1:
        # order of operations
        a, b, c, d = (
            rng.randint(2, 9),
            rng.randint(2, 9),
            rng.randint(2, 9),
            rng.randint(1, 9),
        )
        ans = a + b * c - d
        return (f"Evaluate: {a} + {b} × {c} - {d}", str(ans))
    if kind == 2:
        # average of integers that is integer
        n = rng.randint(3, 5)
        mean = rng.randint(4, 20)
        vals = [mean] * n
        # perturb while keeping sum fixed
        for _ in range(n - 1):
            i, j = rng.sample(range(n), 2)
            delta = rng.randint(1, 3)
            if vals[i] - delta >= 1:
                vals[i] -= delta
                vals[j] += delta
        ans = sum(vals) // n
        listed = ", ".join(str(v) for v in vals)
        return (f"What is the average of these numbers: {listed}?", str(ans))
    # sequential ages / rates style
    hours = rng.randint(2, 6)
    rate = rng.randint(15, 45)
    ans = hours * rate
    return (
        f"A car travels at {rate} miles per hour for {hours} hours. "
        f"How many miles does it travel?",
        str(ans),
    )


def generate_numeric_row(
    *,
    seed: int,
    index: int,
    difficulty: str,
    cfg: DataGenConfig,
) -> dict[str, Any]:
    rng = _stable_rng("numeric", str(seed), difficulty, str(index))
    if difficulty == "easy":
        prompt, answer = _numeric_easy(rng, index)
    elif difficulty == "hard":
        prompt, answer = _numeric_hard(rng, index)
    else:
        prompt, answer = _numeric_medium(rng, index)
    text = prompt + _thinking_suffix(cfg)
    return to_slime_prompt_row(
        text,
        answer,
        rm_type="numeric",
        metadata={
            "benchmark": "gsm8k",
            "stream": "numeric",
            "difficulty": difficulty,
            "task_id": f"numeric_{difficulty}_{seed}_{index}",
            "generator": "seiso.rl_verify.data_gen",
            "seed": seed,
            "index": index,
        },
    )


# ---------------------------------------------------------------------------
# Choice stream
# ---------------------------------------------------------------------------

_CHOICE_BANK: list[tuple[str, list[str], str, str]] = [
    # (stem, options A-D, correct letter, topic)
    (
        "Which planet is known as the Red Planet?",
        ["Venus", "Mars", "Jupiter", "Mercury"],
        "B",
        "science",
    ),
    (
        "What is the chemical symbol for water?",
        ["O2", "CO2", "H2O", "NaCl"],
        "C",
        "science",
    ),
    (
        "How many sides does a hexagon have?",
        ["5", "6", "7", "8"],
        "B",
        "math",
    ),
    (
        "Which number is prime?",
        ["15", "21", "27", "29"],
        "D",
        "math",
    ),
    (
        "What is 2^5?",
        ["10", "16", "25", "32"],
        "D",
        "math",
    ),
    (
        "Which data structure is FIFO?",
        ["Stack", "Queue", "Tree", "Graph"],
        "B",
        "cs",
    ),
    (
        "Which operator has higher precedence in arithmetic?",
        ["+", "×", "both equal", "depends on language"],
        "B",
        "math",
    ),
    (
        "Binary 1011 equals which decimal value?",
        ["9", "10", "11", "13"],
        "C",
        "cs",
    ),
]


def _permute_choice(
    rng: random.Random,
    options: list[str],
    correct_letter: str,
) -> tuple[list[str], str]:
    letters = ["A", "B", "C", "D"]
    correct_idx = letters.index(correct_letter.upper())
    correct_text = options[correct_idx]
    order = list(range(len(options)))
    rng.shuffle(order)
    new_opts = [options[i] for i in order]
    new_letter = letters[new_opts.index(correct_text)]
    return new_opts, new_letter


def generate_choice_row(
    *,
    seed: int,
    index: int,
    difficulty: str,
    cfg: DataGenConfig,
) -> dict[str, Any]:
    rng = _stable_rng("choice", str(seed), difficulty, str(index))
    stem, options, letter, topic = _CHOICE_BANK[index % len(_CHOICE_BANK)]
    # Difficulty modulates distractor rewrites slightly for variety.
    opts = list(options)
    if difficulty == "hard" and topic == "math":
        # Occasionally swap in a near-miss numeric distractor.
        for i, opt in enumerate(opts):
            if opt.isdigit() and rng.random() < 0.4:
                opts[i] = str(int(opt) + rng.choice([-2, -1, 1, 2]))
                # keep correct letter's value intact after permute source
        # restore correct from bank
        correct_idx = ["A", "B", "C", "D"].index(letter)
        opts[correct_idx] = options[correct_idx]
    opts, letter = _permute_choice(rng, opts, letter)
    lines = [f"{stem}"]
    for lab, text in zip(["A", "B", "C", "D"], opts, strict=True):
        lines.append(f"{lab}) {text}")
    lines.append("Reply with the letter of the correct choice only after reasoning.")
    text = "\n".join(lines) + _thinking_suffix(cfg)
    return to_slime_prompt_row(
        text,
        letter,
        rm_type="choice",
        metadata={
            "benchmark": "choice",
            "stream": "choice",
            "difficulty": difficulty,
            "topic": topic,
            "options": opts,
            "task_id": f"choice_{difficulty}_{seed}_{index}",
            "generator": "seiso.rl_verify.data_gen",
            "seed": seed,
            "index": index,
        },
    )


# ---------------------------------------------------------------------------
# Code stream (wraps code_corpus)
# ---------------------------------------------------------------------------


def generate_code_row(
    *,
    seed: int,
    index: int,
    difficulty: str,
    cfg: DataGenConfig,
) -> dict[str, Any]:
    from seiso.rl_verify.code_corpus import generate_grounded_task

    task = generate_grounded_task(
        seed=seed,
        index=index,
        tier=difficulty,
    )
    content = task.prompt
    if cfg.require_thinking_trace and "<think>" not in content.lower():
        content = content.rstrip() + _thinking_suffix(cfg)
    tests = task.tests()
    return to_slime_prompt_row(
        content,
        "",  # code tasks are verified via unit tests, not a scalar label
        rm_type="code",
        metadata={
            "benchmark": "code",
            "stream": "code",
            "difficulty": difficulty,
            "task_id": task.task_id,
            "tests": tests,
            "solution": task.full_source(),
            "timeout_s": task.timeout_s,
            "tags": list(task.tags),
            "generator": "seiso.rl_verify.data_gen",
            "seed": seed,
            "index": index,
        },
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _pick_stream(rng: random.Random, mix: dict[str, float]) -> str:
    streams = list(mix.keys())
    weights = [mix[s] for s in streams]
    return rng.choices(streams, weights=weights, k=1)[0]


def _pick_difficulty(rng: random.Random, mix: dict[str, float]) -> str:
    tiers = list(mix.keys())
    weights = [mix[t] for t in tiers]
    return rng.choices(tiers, weights=weights, k=1)[0]


def generate_rl_corpus(config: DataGenConfig | None = None) -> DataGenResult:
    """Generate a mixed verifiable corpus for online GRPO rollouts.

    Rows are **prompts + labels/tests only**. Completions must be produced by
    the model's data_gen / SGLang rollout path — not stored in this corpus.
    """
    cfg = config or DataGenConfig()
    if cfg.count < 1:
        raise ValueError("count must be at least 1 for meaningful RL data gen")
    stream_mix = parse_weight_mix(
        cfg.mix, allowed=_STREAMS, default=_DEFAULT_STREAM_MIX
    )
    diff_mix = parse_weight_mix(
        cfg.difficulty, allowed=_DIFFICULTIES, default=_DEFAULT_DIFFICULTY_MIX
    )

    plan_rng = _stable_rng("plan", str(cfg.seed), str(cfg.count), json.dumps(stream_mix, sort_keys=True))
    rows: list[dict[str, Any]] = []
    stream_counts: dict[str, int] = {s: 0 for s in stream_mix}
    difficulty_counts: dict[str, int] = {d: 0 for d in diff_mix}

    # Per-stream index counters for stable sub-seeds.
    stream_index = {s: 0 for s in _STREAMS}
    attempts = 0
    max_attempts = max(cfg.count * 10, cfg.count + 50)

    while len(rows) < cfg.count and attempts < max_attempts:
        attempts += 1
        stream = _pick_stream(plan_rng, stream_mix)
        difficulty = _pick_difficulty(plan_rng, diff_mix)
        idx = stream_index[stream]
        stream_index[stream] = idx + 1
        try:
            if stream == "numeric":
                row = generate_numeric_row(
                    seed=cfg.seed, index=idx, difficulty=difficulty, cfg=cfg
                )
            elif stream == "choice":
                row = generate_choice_row(
                    seed=cfg.seed, index=idx, difficulty=difficulty, cfg=cfg
                )
            else:
                row = generate_code_row(
                    seed=cfg.seed, index=idx, difficulty=difficulty, cfg=cfg
                )
                if cfg.verify_code:
                    from seiso.rl_verify.code_proof import verify_code_proof

                    solution = str(row.get("solution") or "")
                    proof = verify_code_proof(
                        solution
                        if "```" in solution
                        else f"```python\n{solution}\n```",
                        row,
                    )
                    if not proof.passed:
                        continue
        except Exception:
            # Skip flaky/invalid generator draws; fail closed on that item only.
            continue

        if "prompt" not in row:
            continue
        rows.append(row)
        stream_counts[stream] = stream_counts.get(stream, 0) + 1
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

    if len(rows) < max(1, cfg.count // 2):
        raise RuntimeError(
            f"data_gen produced only {len(rows)}/{cfg.count} rows after "
            f"{attempts} attempts; check mix/difficulty settings"
        )

    return DataGenResult(
        rows=rows,
        stream_counts=stream_counts,
        difficulty_counts=difficulty_counts,
        seed=cfg.seed,
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


def materialize_rl_corpus(
    out_path: Path,
    config: DataGenConfig | None = None,
    *,
    write_manifest: bool = True,
) -> DataGenResult:
    """Generate corpus and write JSONL (+ optional ``.manifest.json``)."""
    result = generate_rl_corpus(config)
    n = write_jsonl(out_path, result.rows)
    if write_manifest:
        manifest = {
            **result.summary(),
            "path": str(out_path),
            "written": n,
        }
        manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
        if out_path.suffix == ".jsonl":
            manifest_path = out_path.with_name(out_path.stem + ".manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def materialize_for_slime(
    *,
    output_dir: Path,
    count: int = 500,
    seed: int = 0,
    mix: str | dict[str, float] = "numeric:0.5,choice:0.2,code:0.3",
    difficulty: str | dict[str, float] = "easy:0.35,medium:0.45,hard:0.20",
    require_thinking_trace: bool = True,
    thinking_instruction: str | None = None,
    filename: str = "slime_generated.jsonl",
) -> Path:
    """Convenience: write a training corpus under ``output_dir`` and return path."""
    cfg_kwargs: dict[str, Any] = {
        "count": count,
        "seed": seed,
        "mix": mix,
        "difficulty": difficulty,
        "require_thinking_trace": require_thinking_trace,
    }
    if thinking_instruction:
        cfg_kwargs["thinking_instruction"] = thinking_instruction
    cfg = DataGenConfig(**cfg_kwargs)
    path = Path(output_dir) / filename
    materialize_rl_corpus(path, cfg)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "High-level verifiable RL data generation for slime GRPO. "
            "Produces prompts with checkable answers/tests — completions come "
            "from online data_gen / SGLang rollouts, not this file."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/slime_generated.jsonl"),
        help="Output JSONL path (default: data/slime_generated.jsonl)",
    )
    parser.add_argument("--count", type=int, default=500, help="Number of prompts")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed")
    parser.add_argument(
        "--mix",
        type=str,
        default="numeric:0.5,choice:0.2,code:0.3",
        help="Stream mix: numeric / choice / code",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="easy:0.35,medium:0.45,hard:0.20",
        help="Difficulty mix: easy / medium / hard",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Do not append thinking-format instructions to prompts",
    )
    parser.add_argument(
        "--no-verify-code",
        action="store_true",
        help="Skip sandbox verification for code stream (faster, less safe)",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print JSON summary to stdout after write",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = DataGenConfig(
        count=args.count,
        seed=args.seed,
        mix=args.mix,
        difficulty=args.difficulty,
        require_thinking_trace=not args.no_thinking,
        verify_code=not args.no_verify_code,
    )
    result = materialize_rl_corpus(args.out, cfg)
    summary = result.summary()
    summary["path"] = str(args.out)
    if args.print_summary:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"Wrote {summary['count']} prompts → {args.out} "
            f"(streams={summary['stream_counts']}, "
            f"difficulty={summary['difficulty_counts']}, "
            f"answer_diversity={summary['answer_diversity']:.2f})"
        )
    if summary["count"] < 50:
        print(
            "warning: small corpus; for meaningful GRPO prefer --count 200+",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
