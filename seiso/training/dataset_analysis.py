"""Research-grade dataset analysis for dataset-specialized training."""

from __future__ import annotations

import logging
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from seiso.training.config import DatasetFormat
from seiso.training.datasets import detect_format, load_training_dataset
from seiso.training.practices import warmup_ratio_for_corpus
from seiso.training.preprocess import preprocess_training_dataset

logger = logging.getLogger(__name__)

_DOMAIN_LABELS: dict[str, str] = {
    "instruction_tuning": "Instruction tuning (prompt → completion pairs)",
    "conversational": "Multi-turn conversational supervision",
    "preference_alignment": "Preference / chosen-response alignment",
    "causal_lm": "Causal language modeling (plain text)",
    "code_pretraining": "Code / technical corpus pretraining",
    "qa": "Question-answer supervision",
    "unknown": "General supervised fine-tuning",
}

_CODE_COLUMNS = frozenset(
    {"code", "content", "file_content", "raw_content", "programming_language", "language", "lang"}
)
_MATH_COLUMNS = frozenset({"question", "answer", "query", "response", "solution", "problem"})


def _stratified_indices(n: int, *, max_samples: int = 48, seed: int = 42) -> list[int]:
    if n <= 0:
        return []
    if n <= max_samples:
        return list(range(n))
    rng = random.Random(seed)
    anchors = {0, n - 1, n // 2}
    step = max(1, n // max_samples)
    spaced = list(range(0, n, step))[: max_samples - len(anchors)]
    indices = sorted(anchors | set(spaced))
    if len(indices) > max_samples:
        indices = sorted(rng.sample(indices, max_samples))
    return indices


def detect_format_consensus(
    samples: list[dict[str, Any]],
    *,
    forced: DatasetFormat = DatasetFormat.AUTO,
) -> tuple[DatasetFormat, float, dict[str, Any]]:
    """Vote on dataset format from multiple rows (robust to malformed headers)."""
    if forced != DatasetFormat.AUTO:
        return forced, 1.0, {"votes": {forced.value: len(samples)}, "method": "user_override"}

    if not samples:
        return DatasetFormat.TEXT, 0.0, {"votes": {}, "method": "empty"}

    votes: Counter[str] = Counter()
    for sample in samples:
        votes[detect_format(sample).value] += 1

    winner, count = votes.most_common(1)[0]
    confidence = count / max(1, len(samples))
    return DatasetFormat(winner), confidence, {"votes": dict(votes), "method": "schema_vote"}


def _infer_domain(fmt: DatasetFormat, columns: list[str]) -> tuple[str, str]:
    """Classify supervision type from detected schema columns only."""
    colset = {c.lower() for c in columns}

    if fmt == DatasetFormat.TEXT and colset & _CODE_COLUMNS:
        return "code_pretraining", _DOMAIN_LABELS["code_pretraining"]
    if fmt == DatasetFormat.PREFERENCE:
        return "preference_alignment", _DOMAIN_LABELS["preference_alignment"]
    if fmt in (DatasetFormat.CHAT, DatasetFormat.SHAREGPT):
        return "conversational", _DOMAIN_LABELS["conversational"]
    if fmt == DatasetFormat.ALPACA:
        if colset & _MATH_COLUMNS:
            return "qa", _DOMAIN_LABELS["qa"]
        return "instruction_tuning", _DOMAIN_LABELS["instruction_tuning"]
    if fmt == DatasetFormat.TEXT:
        return "causal_lm", _DOMAIN_LABELS["causal_lm"]
    return "unknown", _DOMAIN_LABELS["unknown"]


def _row_char_length(row: dict[str, Any]) -> int:
    return sum(len(str(v)) for v in row.values())


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(pct * len(ordered)) - 1)))
    return ordered[idx]


def _length_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [_row_char_length(row) for row in rows]
    if not lengths:
        return {
            "chars_min": 0,
            "chars_p50": 0,
            "chars_p95": 0,
            "chars_max": 0,
            "estimated_tokens_p95": 0,
        }
    p95 = _percentile(lengths, 0.95)
    return {
        "chars_min": min(lengths),
        "chars_p50": _percentile(lengths, 0.5),
        "chars_p95": p95,
        "chars_max": max(lengths),
        "estimated_tokens_p95": max(1, p95 // 4),
    }


def _recommend_max_seq(length_stats: dict[str, Any]) -> int:
    est = int(length_stats.get("estimated_tokens_p95") or 0)
    if est <= 0:
        return 2048
    padded = int(math.ceil(est * 1.15 / 256) * 256)
    return max(512, min(8192, padded))


def _recommend_epochs(kept: int, domain: str) -> int:
    if kept <= 0:
        return 1
    if domain == "code_pretraining" and kept >= 100_000:
        return 1
    if kept < 500:
        return 8
    if kept < 2_000:
        return 5
    if kept < 20_000:
        return 3
    if kept < 200_000:
        return 2
    return 1


def _recommend_train_on_responses(fmt: DatasetFormat, domain: str) -> bool:
    if fmt == DatasetFormat.TEXT or domain == "code_pretraining":
        return False
    return True


def _preview_rows(rows: list[dict[str, Any]], *, limit: int = 3) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for row in rows[:limit]:
        clipped: dict[str, Any] = {}
        for key, value in row.items():
            text = str(value)
            clipped[key] = text if len(text) <= 240 else text[:237] + "..."
        preview.append(clipped)
    return preview


def build_dataset_training_config(
    *,
    resolved_format: DatasetFormat,
    domain: str,
    kept: int,
    length_stats: dict[str, Any],
) -> dict[str, Any]:
    """Training knobs derived from dataset analysis (not chat/inference defaults)."""
    max_seq = _recommend_max_seq(length_stats)
    epochs = _recommend_epochs(kept, domain)
    train_on_responses_only = _recommend_train_on_responses(resolved_format, domain)
    return {
        "dataset_format": resolved_format.value,
        "train_on_responses_only": train_on_responses_only,
        "preprocess_dataset": True,
        "deduplicate_dataset": True,
        "max_seq_length": max_seq,
        "epochs": epochs,
        "warmup_ratio": warmup_ratio_for_corpus(kept),
        "early_stopping": kept >= 200,
        "early_stopping_patience": 3,
        "packing": resolved_format == DatasetFormat.TEXT and kept >= 10_000,
    }


def analyze_training_dataset(
    dataset: str | Path,
    *,
    dataset_format: DatasetFormat = DatasetFormat.AUTO,
    sandbox_root: Path | None = None,
    split: str = "train",
    sample_rows_for_schema: int = 48,
) -> dict[str, Any]:
    """Analyze the full dataset schema, normalize every row, and recommend training settings.

    Unlike chat/inference context, this inspects the actual training corpus end-to-end so
    preprocessing stats and hyperparameter hints reflect the entire dataset.
    """
    raw = load_training_dataset(str(dataset), split=split, sandbox_root=sandbox_root)
    initial = len(raw)
    if initial == 0:
        raise ValueError("Dataset contains no rows")

    indices = _stratified_indices(initial, max_samples=sample_rows_for_schema)
    schema_samples = [raw[i] for i in indices]
    columns = list(raw.column_names)
    inferred_fmt, confidence, vote_meta = detect_format_consensus(
        schema_samples,
        forced=dataset_format,
    )

    cleaned, stats, resolved_fmt = preprocess_training_dataset(
        raw,
        dataset_format=inferred_fmt if dataset_format == DatasetFormat.AUTO else dataset_format,
        deduplicate=True,
        min_chars=1,
    )

    domain, domain_label = _infer_domain(resolved_fmt, columns)
    length_sample_idx = _stratified_indices(len(cleaned), max_samples=512)
    length_rows = [
        {k: v for k, v in cleaned[i].items() if not str(k).startswith("_")}
        for i in length_sample_idx
    ]
    length_stats = _length_stats(length_rows)
    recommended = build_dataset_training_config(
        resolved_format=resolved_fmt,
        domain=domain,
        kept=stats["kept"],
        length_stats=length_stats,
    )

    utilization = round(100.0 * stats["kept"] / max(1, stats["initial_samples"]), 2)
    notes = [
        f"Scanned all {stats['initial_samples']:,} rows — {stats['kept']:,} trainable after normalization ({utilization}% retained).",
        f"Detected {resolved_fmt.value} schema ({confidence:.0%} vote confidence across stratified samples).",
        f"Domain: {domain_label}.",
        "Training settings below are derived from this dataset only (not chat/inference context).",
    ]
    if stats["removed_invalid"]:
        notes.append(
            f"Dropped {stats['removed_invalid']:,} rows with empty or unparseable supervision targets."
        )
    if stats["removed_duplicate"]:
        notes.append(f"Removed {stats['removed_duplicate']:,} exact duplicate rows.")

    return {
        "valid": stats["kept"] > 0,
        "dataset": str(dataset),
        "split": split,
        "columns": columns,
        "initial_samples": stats["initial_samples"],
        "kept": stats["kept"],
        "removed_invalid": stats["removed_invalid"],
        "removed_duplicate": stats["removed_duplicate"],
        "utilization_pct": utilization,
        "resolved_format": resolved_fmt.value,
        "format_confidence": round(confidence, 4),
        "format_detection": vote_meta,
        "domain": domain,
        "domain_label": domain_label,
        "length_stats": length_stats,
        "recommended_config": recommended,
        "notes": notes,
        "sample_preview": _preview_rows(
            [{k: v for k, v in cleaned[i].items() if not str(k).startswith("_")} for i in range(min(3, len(cleaned)))],
        ),
        "uses_full_dataset": True,
    }


def analysis_notes_for_recommendations(analysis: dict[str, Any]) -> list[str]:
    return list(analysis.get("notes") or [])