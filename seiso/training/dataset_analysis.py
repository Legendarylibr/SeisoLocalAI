"""Research-grade dataset analysis for dataset-specialized training."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from seiso.training.config import DatasetFormat
from seiso.training.datasets import detect_format, load_training_dataset
from seiso.training.practices import warmup_ratio_for_corpus
from seiso.training.preprocess import normalize_sample, preprocess_training_dataset

logger = logging.getLogger(__name__)

_DOMAIN_LABELS: dict[str, str] = {
    "instruction_tuning": "Instruction tuning (prompt → completion pairs)",
    "conversational": "Multi-turn conversational supervision",
    "preference_pairs": (
        "Preference pairs (use Distill-RL DPO; not SFT unless preference_as_sft)"
    ),
    "preference_chosen_sft": (
        "Chosen-response SFT (explicit preference_as_sft; rejected discarded — not DPO)"
    ),
    # Legacy alias kept for older analysis payloads / tests.
    "preference_alignment": (
        "Preference pairs (use Distill-RL DPO; not SFT unless preference_as_sft)"
    ),
    "causal_lm": "Causal language modeling (plain text)",
    "code_pretraining": "Code / technical corpus pretraining",
    "qa": "Question-answer supervision",
    "unknown": "General supervised fine-tuning",
}

_CODE_COLUMNS = frozenset(
    {
        "code",
        "content",
        "file_content",
        "raw_content",
        "programming_language",
        "language",
        "lang",
    }
)
_MATH_COLUMNS = frozenset(
    {"question", "answer", "query", "response", "solution", "problem"}
)


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
        return (
            forced,
            1.0,
            {"votes": {forced.value: len(samples)}, "method": "user_override"},
        )

    if not samples:
        return DatasetFormat.TEXT, 0.0, {"votes": {}, "method": "empty"}

    votes: Counter[str] = Counter()
    for sample in samples:
        votes[detect_format(sample).value] += 1

    winner, count = votes.most_common(1)[0]
    confidence = count / max(1, len(samples))
    return (
        DatasetFormat(winner),
        confidence,
        {"votes": dict(votes), "method": "schema_vote"},
    )


def _infer_domain(fmt: DatasetFormat, columns: list[str]) -> tuple[str, str]:
    """Classify supervision type from detected schema columns only."""
    colset = {c.lower() for c in columns}

    if fmt == DatasetFormat.TEXT and colset & _CODE_COLUMNS:
        return "code_pretraining", _DOMAIN_LABELS["code_pretraining"]
    if fmt == DatasetFormat.PREFERENCE:
        return "preference_pairs", _DOMAIN_LABELS["preference_pairs"]
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
    return not (fmt == DatasetFormat.TEXT or domain == "code_pretraining")


def _preview_rows(
    rows: list[dict[str, Any]], *, limit: int = 3
) -> list[dict[str, Any]]:
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
    packing = resolved_format == DatasetFormat.TEXT and kept >= 10_000
    if packing and train_on_responses_only:
        packing = False
    cfg: dict[str, Any] = {
        "dataset_format": resolved_format.value,
        "train_on_responses_only": train_on_responses_only,
        "preprocess_dataset": True,
        "deduplicate_dataset": True,
        "max_seq_length": max_seq,
        "epochs": epochs,
        "warmup_ratio": warmup_ratio_for_corpus(kept),
        "early_stopping": kept >= 200,
        "early_stopping_patience": 3,
        "packing": packing,
        "preference_as_sft": False,
    }
    if domain in {"preference_pairs", "preference_alignment"} or (
        resolved_format == DatasetFormat.PREFERENCE
    ):
        cfg["dataset_format"] = DatasetFormat.PREFERENCE.value
        cfg["preference_as_sft"] = False
        cfg["packing"] = False
    return cfg


# Process-local cleaned datasets from the most recent analysis (trainer can reuse).
_CLEANED_DATASET_CACHE: dict[str, tuple[Any, dict[str, Any], DatasetFormat]] = {}
_CLEANED_DATASET_CACHE_MAX = 8


def cleaned_dataset_cache_key(
    dataset: str | Path,
    *,
    dataset_format: DatasetFormat,
    sandbox_root: Path | None,
    deduplicate: bool,
    min_chars: int,
) -> str:
    return "|".join(
        [
            str(dataset),
            dataset_format.value,
            str(sandbox_root or ""),
            f"dedupe={int(deduplicate)}",
            f"min_chars={int(min_chars)}",
        ]
    )


def store_cleaned_dataset(
    key: str,
    cleaned: Any,
    stats: dict[str, Any],
    resolved_fmt: DatasetFormat,
) -> None:
    if len(_CLEANED_DATASET_CACHE) >= _CLEANED_DATASET_CACHE_MAX and key not in (
        _CLEANED_DATASET_CACHE
    ):
        _CLEANED_DATASET_CACHE.pop(next(iter(_CLEANED_DATASET_CACHE)))
    _CLEANED_DATASET_CACHE[key] = (cleaned, stats, resolved_fmt)


def take_cleaned_dataset(
    key: str,
) -> tuple[Any, dict[str, Any], DatasetFormat] | None:
    """Pop a cached cleaned dataset (one-shot reuse for the following train job)."""
    return _CLEANED_DATASET_CACHE.pop(key, None)


def _analysis_payload(
    *,
    dataset: str | Path,
    split: str,
    columns: list[str],
    initial_samples: int,
    kept: int,
    removed_invalid: int,
    removed_duplicate: int,
    resolved_fmt: DatasetFormat,
    confidence: float,
    vote_meta: dict[str, Any],
    domain: str,
    domain_label: str,
    length_stats: dict[str, Any],
    notes: list[str],
    sample_preview: list[dict[str, Any]],
    uses_full_dataset: bool,
    cleaned_cache_key: str | None = None,
) -> dict[str, Any]:
    utilization = round(100.0 * kept / max(1, initial_samples), 2)
    recommended = build_dataset_training_config(
        resolved_format=resolved_fmt,
        domain=domain,
        kept=kept,
        length_stats=length_stats,
    )
    payload: dict[str, Any] = {
        "valid": kept > 0,
        "dataset": str(dataset),
        "split": split,
        "columns": columns,
        "initial_samples": initial_samples,
        "kept": kept,
        "removed_invalid": removed_invalid,
        "removed_duplicate": removed_duplicate,
        "utilization_pct": utilization,
        "resolved_format": resolved_fmt.value,
        "format_confidence": round(confidence, 4),
        "format_detection": vote_meta,
        "domain": domain,
        "domain_label": domain_label,
        "length_stats": length_stats,
        "recommended_config": recommended,
        "notes": notes,
        "sample_preview": sample_preview,
        "uses_full_dataset": uses_full_dataset,
        "preprocess_defaults": {"deduplicate": True, "min_chars": 1},
    }
    if cleaned_cache_key is not None:
        payload["cleaned_cache_key"] = cleaned_cache_key
    return payload


def _analyze_sampled(
    *,
    dataset: str | Path,
    split: str,
    initial: int,
    columns: list[str],
    schema_samples: list[dict[str, Any]],
    inferred_fmt: DatasetFormat,
    confidence: float,
    vote_meta: dict[str, Any],
    dataset_format: DatasetFormat,
) -> dict[str, Any]:
    """Cheap validation path: normalize stratified samples only (no full map)."""
    fmt = inferred_fmt if dataset_format == DatasetFormat.AUTO else dataset_format
    cleaned_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    removed_invalid = 0
    removed_duplicate = 0
    for sample in schema_samples:
        row = normalize_sample(sample, fmt)
        if row is None:
            removed_invalid += 1
            continue
        payload = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
        fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if fingerprint in seen:
            removed_duplicate += 1
            continue
        seen.add(fingerprint)
        cleaned_rows.append(row)

    if not cleaned_rows:
        raise ValueError(
            "Dataset sample has no trainable rows after normalization "
            f"(format={fmt.value})"
        )

    sample_n = max(1, len(schema_samples))
    kept_ratio = len(cleaned_rows) / sample_n
    estimated_kept = max(1, min(initial, int(round(kept_ratio * initial))))
    estimated_invalid = max(
        0, min(initial - estimated_kept, int(round(removed_invalid / sample_n * initial)))
    )
    estimated_duplicate = max(
        0,
        min(
            initial - estimated_kept - estimated_invalid,
            int(round(removed_duplicate / sample_n * initial)),
        ),
    )

    domain, domain_label = _infer_domain(fmt, columns)
    length_stats = _length_stats(cleaned_rows)
    notes = [
        f"Sampled {sample_n:,}/{initial:,} rows for quick validation — "
        f"estimated ~{estimated_kept:,} trainable after normalization "
        f"({round(100.0 * estimated_kept / max(1, initial), 2)}% retained).",
        f"Detected {fmt.value} schema ({confidence:.0%} vote confidence across stratified samples).",
        f"Domain: {domain_label}.",
        "Full preprocess/dedupe runs once during training (not duplicated here).",
        "Training settings below are derived from this dataset sample (not chat/inference context).",
    ]
    if fmt == DatasetFormat.PREFERENCE:
        notes.append(
            "Preference pairs require Distill-RL/DPO (`seiso distill-rl`) for real "
            "alignment. SFT refuses this dataset unless preference_as_sft=true."
        )
    if removed_invalid:
        notes.append(
            f"Sample dropped {removed_invalid:,} rows with empty or unparseable supervision targets."
        )
    if removed_duplicate:
        notes.append(f"Sample removed {removed_duplicate:,} exact duplicate rows.")

    return _analysis_payload(
        dataset=dataset,
        split=split,
        columns=columns,
        initial_samples=initial,
        kept=estimated_kept,
        removed_invalid=estimated_invalid,
        removed_duplicate=estimated_duplicate,
        resolved_fmt=fmt,
        confidence=confidence,
        vote_meta=vote_meta,
        domain=domain,
        domain_label=domain_label,
        length_stats=length_stats,
        notes=notes,
        sample_preview=_preview_rows(cleaned_rows),
        uses_full_dataset=False,
    )


def analyze_training_dataset(
    dataset: str | Path,
    *,
    dataset_format: DatasetFormat = DatasetFormat.AUTO,
    sandbox_root: Path | None = None,
    split: str = "train",
    sample_rows_for_schema: int = 48,
    full_scan: bool = True,
) -> dict[str, Any]:
    """Analyze dataset schema and recommend training settings.

    When ``full_scan`` is True (default, used by the UI analyze endpoint), every
    row is normalized and deduplicated so stats reflect the full corpus. The
    cleaned dataset is stashed for one-shot reuse by the following train job.

    When ``full_scan`` is False (used at train startup without a prior UI
    analysis), only stratified samples are validated so the expensive full
    preprocess is not repeated before ``_prepare_datasets`` runs it once.
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

    if not full_scan:
        return _analyze_sampled(
            dataset=dataset,
            split=split,
            initial=initial,
            columns=columns,
            schema_samples=schema_samples,
            inferred_fmt=inferred_fmt,
            confidence=confidence,
            vote_meta=vote_meta,
            dataset_format=dataset_format,
        )

    effective_fmt = (
        inferred_fmt if dataset_format == DatasetFormat.AUTO else dataset_format
    )
    # Preference pairs: analyze chosen-side structure with an explicit opt-in for
    # stats only. Do not cache cleaned rows — SFT train must refuse without
    # preference_as_sft (real alignment is Distill-RL/DPO).
    preference_stats_only = effective_fmt == DatasetFormat.PREFERENCE
    cleaned, stats, resolved_fmt = preprocess_training_dataset(
        raw,
        dataset_format=effective_fmt,
        deduplicate=True,
        min_chars=1,
        preference_as_sft=preference_stats_only,
    )
    cache_key = cleaned_dataset_cache_key(
        dataset,
        dataset_format=dataset_format,
        sandbox_root=sandbox_root,
        deduplicate=True,
        min_chars=1,
    )
    if not preference_stats_only:
        store_cleaned_dataset(cache_key, cleaned, stats, resolved_fmt)

    domain_fmt = DatasetFormat.PREFERENCE if preference_stats_only else resolved_fmt
    domain, domain_label = _infer_domain(domain_fmt, columns)
    length_sample_idx = _stratified_indices(len(cleaned), max_samples=512)
    length_rows = [
        {k: v for k, v in cleaned[i].items() if not str(k).startswith("_")}
        for i in length_sample_idx
    ]
    length_stats = _length_stats(length_rows)

    notes = [
        f"Scanned all {stats['initial_samples']:,} rows — {stats['kept']:,} trainable after normalization "
        f"({round(100.0 * stats['kept'] / max(1, stats['initial_samples']), 2)}% retained).",
        f"Detected {domain_fmt.value} schema ({confidence:.0%} vote confidence across stratified samples).",
        f"Domain: {domain_label}.",
        "Training settings below are derived from this dataset only (not chat/inference context).",
    ]
    if preference_stats_only:
        notes.append(
            "Preference pairs require Distill-RL/DPO (`seiso distill-rl`) for real "
            "alignment. Training Studio SFT refuses this dataset unless "
            "preference_as_sft=true (chosen-only SFT; rejected discarded)."
        )
    if stats["removed_invalid"]:
        notes.append(
            f"Dropped {stats['removed_invalid']:,} rows with empty or unparseable supervision targets."
        )
    if stats["removed_duplicate"]:
        notes.append(f"Removed {stats['removed_duplicate']:,} exact duplicate rows.")

    return _analysis_payload(
        dataset=dataset,
        split=split,
        columns=columns,
        initial_samples=stats["initial_samples"],
        kept=stats["kept"],
        removed_invalid=stats["removed_invalid"],
        removed_duplicate=stats["removed_duplicate"],
        resolved_fmt=domain_fmt,
        confidence=confidence,
        vote_meta=vote_meta,
        domain=domain,
        domain_label=domain_label,
        length_stats=length_stats,
        notes=notes,
        sample_preview=_preview_rows(
            [
                {k: v for k, v in cleaned[i].items() if not str(k).startswith("_")}
                for i in range(min(3, len(cleaned)))
            ],
        ),
        uses_full_dataset=True,
        cleaned_cache_key=None if preference_stats_only else cache_key,
    )


def analysis_notes_for_recommendations(analysis: dict[str, Any]) -> list[str]:
    return list(analysis.get("notes") or [])
