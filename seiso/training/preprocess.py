"""Normalize and clean training datasets so every valid row is used effectively."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from seiso.training.config import DatasetFormat
from seiso.training.datasets import detect_format

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_HUMAN_ASSISTANT_TURN_RE = re.compile(
    r"(?:^|\n)\s*(Human|Assistant)\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)


def _normalize_dialog_text(text: Any) -> str:
    """Normalize line endings without collapsing turn boundaries."""
    if text is None:
        return ""
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    return normalized


def parse_human_assistant_dialog(text: Any) -> list[dict[str, str]]:
    """Parse Human/Assistant dialog transcripts into chat messages."""
    raw = _normalize_dialog_text(text)
    if not raw:
        return []

    matches = list(_HUMAN_ASSISTANT_TURN_RE.finditer(raw))
    if not matches:
        return []

    messages: list[dict[str, str]] = []
    for idx, match in enumerate(matches):
        role_raw = match.group(1).lower()
        role = "user" if role_raw == "human" else "assistant"
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        content = raw[start:end].strip()
        content = _WHITESPACE_RE.sub(" ", content).strip()
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _strip_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return _WHITESPACE_RE.sub(" ", text).strip()


def _normalize_role(role: str) -> str:
    r = role.strip().lower()
    if r in ("human", "user"):
        return "user"
    if r in ("gpt", "assistant", "bot", "model"):
        return "assistant"
    if r in ("system", "tool"):
        return r
    return "user"


def normalize_sample(
    sample: dict[str, Any], fmt: DatasetFormat
) -> dict[str, Any] | None:
    """Map a raw row to a canonical schema, or None if it has no trainable content."""
    if fmt == DatasetFormat.TEXT:
        text = _strip_text(sample.get("text") or sample.get("content"))
        if len(text) < 1:
            return None
        return {"text": text}

    if fmt == DatasetFormat.ALPACA:
        if "query" in sample and "response" in sample:
            query = _strip_text(sample.get("query"))
            response = _strip_text(sample.get("response"))
            if not response:
                return None
            return {"query": query, "response": response}
        if "question" in sample and "answer" in sample:
            question = _strip_text(sample.get("question"))
            answer = _strip_text(sample.get("answer"))
            if not answer:
                return None
            return {"question": question, "answer": answer}
        if "prompt" in sample and ("completion" in sample or "response" in sample):
            prompt = _strip_text(sample.get("prompt"))
            completion = _strip_text(sample.get("completion") or sample.get("response"))
            if not completion:
                return None
            return {"instruction": prompt, "output": completion}
        instruction = _strip_text(sample.get("instruction"))
        inp = _strip_text(sample.get("input"))
        output = _strip_text(sample.get("output") or sample.get("response"))
        if not output:
            return None
        if not instruction and not inp:
            return None
        row: dict[str, Any] = {"instruction": instruction, "output": output}
        if inp:
            row["input"] = inp
        return row

    if fmt == DatasetFormat.SHAREGPT and "conversations" in sample:
        turns: list[dict[str, str]] = []
        for turn in sample["conversations"]:
            content = _strip_text(turn.get("value") or turn.get("content"))
            if not content:
                continue
            role = _normalize_role(str(turn.get("from") or turn.get("role") or "user"))
            from_role = (
                "human" if role == "user" else "gpt" if role == "assistant" else role
            )
            turns.append({"from": from_role, "value": content})
        if not turns or not any(t["from"] == "gpt" for t in turns):
            return None
        return {"conversations": turns}

    if fmt == DatasetFormat.PREFERENCE:
        chosen = (
            sample.get("chosen")
            or sample.get("chosen_response")
            or sample.get("accepted")
        )
        messages: list[dict[str, str]] = parse_human_assistant_dialog(chosen)
        if not messages:
            prompt = _strip_text(sample.get("prompt"))
            response = _strip_text(
                sample.get("chosen") or sample.get("chosen_response")
            )
            if (
                prompt
                and response
                and "Human:" not in response
                and "Assistant:" not in response
            ):
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
        if not messages or not any(m["role"] == "assistant" for m in messages):
            return None
        return {"messages": messages}

    if fmt == DatasetFormat.CHAT or "messages" in sample:
        messages = []
        for turn in sample.get("messages") or []:
            content = _strip_text(turn.get("content"))
            if not content:
                continue
            messages.append(
                {
                    "role": _normalize_role(str(turn.get("role") or "user")),
                    "content": content,
                }
            )
        if not messages or not any(m["role"] == "assistant" for m in messages):
            return None
        return {"messages": messages}

    text = _strip_text(sample.get("text") or sample.get("content") or "")
    if not text:
        return None
    return {"text": text}


def _content_fingerprint(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_eval_split_size(
    n: int,
    eval_split_ratio: float,
    max_eval_samples: int,
    *,
    min_train_samples: int = 1,
) -> int:
    """Cap validation size so the majority of rows stay in the training split."""
    if n <= 10 or eval_split_ratio <= 0:
        return 0
    eval_n = max(1, int(n * eval_split_ratio))
    eval_n = min(eval_n, max_eval_samples, n - min_train_samples)
    return max(0, eval_n)


def _first_unique_indices(fingerprints) -> list[int]:
    """Return sorted indices of first occurrences (numpy when available)."""
    try:
        import numpy as np

        arr = np.asarray(fingerprints, dtype=object)
        _, first_idx = np.unique(arr, return_index=True)
        return np.sort(first_idx).tolist()
    except Exception:
        seen: set[str] = set()
        keep: list[int] = []
        for idx, key in enumerate(fingerprints):
            if key in seen:
                continue
            seen.add(key)
            keep.append(idx)
        return keep


def _datasets_map_kwargs(num_proc: int | None) -> dict[str, Any]:
    """Map/filter kwargs safe for Forge workers (no tqdm writes to closed pipes)."""
    from datasets.utils.logging import disable_progress_bar

    disable_progress_bar()
    kwargs: dict[str, Any] = {}
    if num_proc and num_proc > 1:
        kwargs["num_proc"] = num_proc
    return kwargs


def preprocess_training_dataset(
    dataset,
    *,
    dataset_format: DatasetFormat = DatasetFormat.AUTO,
    deduplicate: bool = True,
    min_chars: int = 1,
    num_proc: int | None = None,
    preference_as_sft: bool = False,
) -> tuple[Any, dict[str, Any], DatasetFormat]:
    """Normalize rows, drop invalid/empty samples, and optionally deduplicate.

    Preference (chosen/rejected) rows require ``preference_as_sft=True`` to continue
    as chosen-only chat SFT. Otherwise raise — real preference learning is Distill-RL/DPO.
    """
    initial = len(dataset)
    resolved_fmt = dataset_format
    if resolved_fmt == DatasetFormat.AUTO and initial > 0:
        resolved_fmt = detect_format(dataset[0])

    if resolved_fmt == DatasetFormat.PREFERENCE and not preference_as_sft:
        raise ValueError(
            "Preference datasets (chosen/rejected) are not SFT alignment. "
            "Use Distill-RL/DPO (`seiso distill-rl`) for real preference learning, "
            "or set preference_as_sft=true to train supervised on chosen responses "
            "only (rejected pairs are discarded)."
        )

    stats: dict[str, Any] = {
        "initial_samples": initial,
        "resolved_format": resolved_fmt.value,
        "removed_invalid": 0,
        "removed_duplicate": 0,
        "kept": 0,
        "preference_as_sft": bool(
            preference_as_sft and resolved_fmt == DatasetFormat.PREFERENCE
        ),
    }

    def transform(sample: dict[str, Any]) -> dict[str, Any]:
        norm = normalize_sample(sample, resolved_fmt)
        if norm is None:
            return {"_seiso_valid": False}
        if sum(len(str(v)) for v in norm.values()) < min_chars:
            return {"_seiso_valid": False}
        return {
            **norm,
            "_seiso_valid": True,
            "_seiso_fingerprint": _content_fingerprint(norm),
        }

    map_kwargs = _datasets_map_kwargs(num_proc)
    mapped = dataset.map(transform, **map_kwargs)
    before_filter = len(mapped)
    filtered = mapped.filter(lambda row: row["_seiso_valid"], **map_kwargs)
    stats["removed_invalid"] = before_filter - len(filtered)

    if deduplicate and len(filtered) > 0:
        try:
            fingerprints = filtered["_seiso_fingerprint"]
        except (KeyError, TypeError):
            fingerprints = [
                str(filtered[idx].get("_seiso_fingerprint") or "")
                for idx in range(len(filtered))
            ]
        keep_indices = _first_unique_indices(fingerprints)
        if len(keep_indices) < len(filtered):
            stats["removed_duplicate"] = len(filtered) - len(keep_indices)
            filtered = filtered.select(keep_indices)

    drop_cols = [c for c in filtered.column_names if c.startswith("_seiso")]
    final = filtered.remove_columns(drop_cols) if drop_cols else filtered
    stats["kept"] = len(final)

    logger.info(
        "Dataset preprocess: %d -> %d (format=%s, invalid=%d, dupes=%d)",
        initial,
        stats["kept"],
        resolved_fmt.value,
        stats["removed_invalid"],
        stats["removed_duplicate"],
    )
    if stats["kept"] == 0:
        raise ValueError(
            f"No valid training samples after preprocessing "
            f"({stats['removed_invalid']} invalid/empty, {stats['removed_duplicate']} dups). "
            f"Format: {resolved_fmt.value}. "
            "Dataset must contain usable assistant responses or text after normalization."
        )

    if resolved_fmt == DatasetFormat.PREFERENCE:
        # Explicit opt-in: chosen-only SFT (rejected discarded). Not DPO/ORPO.
        resolved_fmt = DatasetFormat.CHAT
        stats["resolved_format"] = resolved_fmt.value
        stats["preference_as_sft"] = True

    return final, stats, resolved_fmt
