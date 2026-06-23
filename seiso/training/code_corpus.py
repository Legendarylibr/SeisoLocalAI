"""Normalize code pretraining corpora from Hugging Face into causal-LM JSONL rows."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TEXT_COLUMNS = ("text", "content", "code", "raw_content", "file_content")
_LANG_COLUMNS = ("language", "lang", "programming_language")
_PATH_COLUMNS = ("rel_path", "path", "file_path", "filename")


@dataclass(frozen=True)
class NormalizedCodeSample:
    text: str
    language: str | None = None
    source: str | None = None

    def to_record(self) -> dict[str, str]:
        return {"text": self.text}


def _first_str(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def normalize_code_row(row: dict[str, Any], *, source: str | None = None) -> NormalizedCodeSample | None:
    """Convert a code-hub row into a single training text field."""
    body = _first_str(row, _TEXT_COLUMNS)
    if not body:
        return None

    language = _first_str(row, _LANG_COLUMNS)
    rel_path = _first_str(row, _PATH_COLUMNS)
    repo = str(row.get("repo") or row.get("repository") or "").strip() or None

    parts: list[str] = []
    if language:
        parts.append(f"# language: {language.lower()}")
    if repo and rel_path:
        parts.append(f"# path: {repo}/{rel_path}")
    elif rel_path:
        parts.append(f"# path: {rel_path}")
    parts.append(body.rstrip())
    text = "\n".join(parts).strip()
    if len(text) < 32:
        return None
    if len(text) > 256_000:
        text = text[:256_000]
    return NormalizedCodeSample(text=text, language=language, source=source or repo)


def recommend_pretraining_epochs(
    *,
    sample_count: int,
    max_seq_length: int,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 32,
    model_params_b: float = 7.0,
    target_tokens_per_param: float = 20.0,
) -> dict[str, Any]:
    """Recommend epoch count for large code corpora on limited hardware."""
    del batch_size, gradient_accumulation_steps
    avg_tokens = int(max_seq_length * 0.65)
    tokens_per_epoch = max(1, sample_count * avg_tokens)
    chinchilla_target = int(model_params_b * 1e9 * target_tokens_per_param)
    epochs_for_chinchilla = max(1, chinchilla_target // tokens_per_epoch)

    if sample_count <= 0:
        return {
            "recommended_epochs": 1,
            "tokens_per_epoch_estimate": 0,
            "chinchilla_target_tokens": chinchilla_target,
            "note": "No samples — prepare normalized JSONL before training.",
        }

    if sample_count >= 1_000_000:
        recommended = 1
        note = (
            "Corpus is web-scale — use 1 epoch to avoid LoRA overfitting; "
            "increase max_samples or run multi-node for more token coverage."
        )
    elif epochs_for_chinchilla <= 1:
        recommended = 1
        note = "Subset fits within one Chinchilla-optimal pass."
    else:
        recommended = min(3, epochs_for_chinchilla)
        note = f"Chinchilla-optimal pass suggests up to {epochs_for_chinchilla} epochs on this subset."

    return {
        "recommended_epochs": recommended,
        "tokens_per_epoch_estimate": tokens_per_epoch,
        "chinchilla_target_tokens": chinchilla_target,
        "note": note,
    }


def write_jsonl(records: Iterator[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def is_metadata_only_row(row: dict[str, Any]) -> bool:
    """True when a row has file metadata but no trainable text body."""
    has_meta = any(row.get(key) for key in ("repo", "rel_path", "commit_id", "commit"))
    has_body = _first_str(row, _TEXT_COLUMNS) is not None
    return bool(has_meta and not has_body)
