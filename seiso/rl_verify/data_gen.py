"""Shared slime JSONL row helpers for grounded RL materialize paths.

Product corpora come from operator/HF JSONL, ``dataset`` prep, or opt-in
Data Designer — not from a local arithmetic/choice toy generator (removed).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataGenResult:
    """Materialized rows plus light diagnostics for GRPO readiness."""

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
            "answer_diversity": (unique_answers / max(1, len(answers)) if answers else 0.0),
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
        raise ValueError(f"unknown mix keys {sorted(unknown)}; allowed: {sorted(allowed)}")
    if not raw:
        raw = dict(default)
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("mix weights must sum to > 0")
    return {k: v / total for k, v in raw.items() if v > 0}


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
    meta.setdefault("source_name", "seiso.rl_verify")
    row: dict[str, Any] = {
        "prompt": [{"role": "user", "content": content}],
        "label": label,
        "answer": label,
        "metadata": meta,
    }
    row["reward"] = rm_type
    row["benchmark"] = meta.get("benchmark", rm_type)
    if "tests" in meta:
        row["tests"] = meta["tests"]
    if "solution" in meta:
        row["solution"] = meta["solution"]
    if "timeout_s" in meta:
        row["timeout_s"] = meta["timeout_s"]
    return row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count
