"""Dataset loading and chat-template formatting for HF training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from seiso.models.chat_format import extract_messages, format_messages_for_prompt
from seiso.training.config import DatasetFormat

logger = logging.getLogger(__name__)


def load_training_dataset(
    path: str | Path, split: str = "train", *, sandbox_root: Path | None = None
):
    """Load dataset from HF hub ID, JSON/JSONL file, or directory."""
    from seiso.security import assert_within

    p = Path(path).expanduser()
    if p.exists() and sandbox_root is not None:
        assert_within(sandbox_root, p)

    from datasets import Dataset, load_dataset

    if p.exists():
        if p.suffix == ".jsonl":
            logger.info("Loading JSONL via datasets mmap (%s)", p)
            return load_dataset("json", data_files=str(p), split=split)
        if p.suffix == ".json":
            # Prefer Arrow mmap for array JSON; fall back for tiny / non-list payloads.
            try:
                return load_dataset("json", data_files=str(p), split=split)
            except Exception:
                data = json.loads(p.read_text())
                return Dataset.from_list(data if isinstance(data, list) else [data])
        if p.is_dir():
            return load_dataset(
                str(p), split=split, revision="main"
            )  # nosec B615: local path, revision pinned for hub fallback
    return load_dataset(
        str(path), split=split, revision="main"
    )  # nosec B615: revision pinned


def detect_format(sample: dict) -> DatasetFormat:
    if ("chosen" in sample and "rejected" in sample) or (
        "chosen" in sample and "prompt" in sample
    ):
        return DatasetFormat.PREFERENCE
    if ("query" in sample and "response" in sample) or (
        "question" in sample and "answer" in sample
    ):
        return DatasetFormat.ALPACA
    if ("prompt" in sample and "completion" in sample) or (
        "prompt" in sample and "response" in sample
    ):
        return DatasetFormat.ALPACA
    if "conversations" in sample or "messages" in sample:
        return (
            DatasetFormat.SHAREGPT if "conversations" in sample else DatasetFormat.CHAT
        )
    if ("instruction" in sample and "output" in sample) or (
        "instruction" in sample and "response" in sample
    ):
        return DatasetFormat.ALPACA
    if "text" in sample or "content" in sample or "code" in sample:
        return DatasetFormat.TEXT
    return DatasetFormat.TEXT


def should_disable_packing_for_response_mask(
    packing: bool,
    train_on_responses_only: bool,
    fmt: DatasetFormat,
) -> bool:
    """Packing is for large text corpora; response-only chat needs Seiso masks."""
    return bool(
        packing and train_on_responses_only and fmt != DatasetFormat.TEXT
    )


def format_sample(sample: dict, fmt: DatasetFormat, tokenizer) -> str:
    """Format a single sample into training text using chat template when available."""
    if fmt == DatasetFormat.TEXT:
        return sample.get("text") or sample.get("content") or str(sample)

    messages = extract_messages(sample, fmt)
    return format_messages_for_prompt(messages, tokenizer, add_generation_prompt=False)


def _build_labels(input_ids: list[int], prompt_len: int) -> list[int]:
    labels = list(input_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100
    return labels


def _rows_from_batch(batch: dict[str, list[Any]]) -> list[dict[str, Any]]:
    return [
        dict(zip(batch.keys(), values, strict=True))
        for values in zip(*batch.values(), strict=True)
    ]


def _tokenizer_supports_assistant_mask(tokenizer) -> bool:
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None:
        return False
    try:
        import inspect

        return "return_assistant_tokens_mask" in inspect.signature(apply).parameters
    except (TypeError, ValueError):
        return False


def _eos_token_id(tokenizer) -> int | None:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, int) and eos_id >= 0:
        return eos_id
    return None


def _encode_chat_ids(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int] | None:
    """Tokenize messages via chat template only (no max-length clip)."""
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None:
        return None
    try:
        ids = apply(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=False,
        )
    except TypeError:
        try:
            ids = apply(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            return None
    except Exception:
        return None
    if not isinstance(ids, list) or not ids:
        return None
    if isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def _labels_from_assistant_mask(
    ids: list[int], mask: list[Any]
) -> list[int] | None:
    if not ids or mask is None:
        return None
    mask_list = list(mask)[: len(ids)]
    if len(mask_list) < len(ids):
        mask_list = mask_list + [0] * (len(ids) - len(mask_list))
    labels = [tok if m else -100 for tok, m in zip(ids, mask_list, strict=False)]
    if all(label == -100 for label in labels):
        return None
    return labels


def _labels_from_assistant_spans(
    tokenizer,
    messages: list[dict[str, Any]],
    full_ids: list[int],
) -> list[int] | None:
    """Unmask every assistant turn using template-tokenized turn boundaries."""
    labels = [-100] * len(full_ids)
    found = False
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        prefix_ids = _encode_chat_ids(
            tokenizer, messages[:i], add_generation_prompt=True
        )
        with_asst = _encode_chat_ids(
            tokenizer, messages[: i + 1], add_generation_prompt=False
        )
        if prefix_ids is None or with_asst is None:
            logger.debug(
                "Assistant span encode failed at turn %d; trying last-turn fallback",
                i,
            )
            return None
        start, end = len(prefix_ids), len(with_asst)
        if start > end or end > len(full_ids):
            logger.debug(
                "Assistant span out of range at turn %d (start=%d end=%d full=%d)",
                i,
                start,
                end,
                len(full_ids),
            )
            return None
        # Prefix of with_asst must match full conversation encoding.
        if with_asst != full_ids[:end]:
            logger.debug(
                "Assistant span not prefix-consistent at turn %d; last-turn fallback",
                i,
            )
            return None
        for j in range(start, end):
            labels[j] = full_ids[j]
        found = True
    if not found or all(label == -100 for label in labels):
        return None
    return labels


def _labels_last_assistant_turn(
    tokenizer,
    messages: list[dict[str, Any]],
    full_ids: list[int],
) -> list[int] | None:
    """Fallback: supervise only the final assistant turn via template lengths."""
    if not messages or messages[-1].get("role") != "assistant":
        return None
    prompt_ids = _encode_chat_ids(
        tokenizer, messages[:-1], add_generation_prompt=True
    )
    if prompt_ids is None:
        return None
    labels = _build_labels(full_ids, len(prompt_ids))
    if all(label == -100 for label in labels):
        return None
    return labels


def _truncate_keep_end(
    ids: list[int],
    labels: list[int],
    attention: list[int],
    max_len: int,
) -> tuple[list[int], list[int], list[int]]:
    if max_len <= 0 or len(ids) <= max_len:
        return ids, labels, attention
    return ids[-max_len:], labels[-max_len:], attention[-max_len:]


def _truncate_keep_start(
    ids: list[int],
    labels: list[int],
    attention: list[int],
    max_len: int,
) -> tuple[list[int], list[int], list[int]]:
    if max_len <= 0 or len(ids) <= max_len:
        return ids, labels, attention
    return ids[:max_len], labels[:max_len], attention[:max_len]


def _ensure_eos(
    ids: list[int],
    labels: list[int],
    attention: list[int],
    eos_id: int | None,
    *,
    max_len: int,
) -> tuple[list[int], list[int], list[int]]:
    if eos_id is None or not ids:
        return ids, labels, attention
    if ids[-1] == eos_id:
        return ids, labels, attention
    if len(ids) < max_len:
        ids = ids + [eos_id]
        # Supervise EOS when the prior token was supervised; else still teach stop.
        labels = labels + [eos_id]
        attention = attention + [1]
        return ids, labels, attention
    # At budget: replace last token with EOS and supervise it.
    ids = list(ids)
    labels = list(labels)
    attention = list(attention)
    ids[-1] = eos_id
    labels[-1] = eos_id
    attention[-1] = 1
    return ids, labels, attention


def _encode_with_assistant_mask_api(
    tokenizer, messages: list[dict[str, Any]]
) -> tuple[list[int], list[int]] | None:
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None or not _tokenizer_supports_assistant_mask(tokenizer):
        return None
    try:
        encoded = apply(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
    except Exception:
        return None
    ids = encoded.get("input_ids")
    mask = encoded.get("assistant_masks") or encoded.get("assistant_mask")
    if ids is None or mask is None:
        return None
    if isinstance(ids[0], list):
        ids = ids[0]
        mask = mask[0]
    ids_list = list(ids)
    labels = _labels_from_assistant_mask(ids_list, list(mask))
    if labels is None:
        return None
    return ids_list, labels


def _tokenize_string_row(
    text: str,
    tokenizer,
    max_seq_length: int,
    *,
    keep_end: bool,
) -> dict[str, list[int]]:
    # Avoid dual truncation args; clip ourselves for keep_end vs keep_start.
    encoded = tokenizer(text, truncation=False, padding=False)
    ids = list(encoded["input_ids"])
    attention = list(encoded.get("attention_mask") or [1] * len(ids))
    labels = list(ids)
    if keep_end:
        ids, labels, attention = _truncate_keep_end(
            ids, labels, attention, max_seq_length
        )
    else:
        ids, labels, attention = _truncate_keep_start(
            ids, labels, attention, max_seq_length
        )
    ids, labels, attention = _ensure_eos(
        ids, labels, attention, _eos_token_id(tokenizer), max_len=max_seq_length
    )
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "labels": labels,
    }


def _tokenize_chat_row(
    sample: dict[str, Any],
    fmt: DatasetFormat,
    tokenizer,
    max_seq_length: int,
    *,
    mask_assistant_only: bool,
) -> dict[str, list[int]]:
    """Encode one chat/alpaca/sharegpt/preference row via chat template when possible."""
    messages = extract_messages(sample, fmt)
    eos_id = _eos_token_id(tokenizer)

    if not messages:
        text = format_sample(sample, fmt, tokenizer)
        return _tokenize_string_row(
            text, tokenizer, max_seq_length, keep_end=mask_assistant_only
        )

    # Prefer a single template encode for input_ids.
    full_ids: list[int] | None = None
    labels: list[int] | None = None

    if mask_assistant_only:
        masked = _encode_with_assistant_mask_api(tokenizer, messages)
        if masked is not None:
            full_ids, labels = masked
        if full_ids is None:
            full_ids = _encode_chat_ids(
                tokenizer, messages, add_generation_prompt=False
            )
        if full_ids is not None and labels is None:
            labels = _labels_from_assistant_spans(tokenizer, messages, full_ids)
        if full_ids is not None and labels is None:
            labels = _labels_last_assistant_turn(tokenizer, messages, full_ids)
    else:
        full_ids = _encode_chat_ids(tokenizer, messages, add_generation_prompt=False)
        if full_ids is not None:
            labels = list(full_ids)

    if full_ids is None or labels is None:
        # No chat template (or encode failure): string path.
        text = format_sample(sample, fmt, tokenizer)
        return _tokenize_string_row(
            text, tokenizer, max_seq_length, keep_end=mask_assistant_only
        )

    attention = [1] * len(full_ids)
    if mask_assistant_only:
        full_ids, labels, attention = _truncate_keep_end(
            full_ids, labels, attention, max_seq_length
        )
    else:
        full_ids, labels, attention = _truncate_keep_start(
            full_ids, labels, attention, max_seq_length
        )

    full_ids, labels, attention = _ensure_eos(
        full_ids, labels, attention, eos_id, max_len=max_seq_length
    )
    return {
        "input_ids": full_ids,
        "attention_mask": attention,
        "labels": labels,
    }


def _tokenize_text_row(
    sample: dict[str, Any],
    tokenizer,
    max_seq_length: int,
) -> dict[str, list[int]]:
    text = sample.get("text") or sample.get("content") or str(sample)
    return _tokenize_string_row(text, tokenizer, max_seq_length, keep_end=False)


def prepare_tokenized_dataset(
    dataset,
    tokenizer,
    *,
    max_seq_length: int,
    dataset_format: DatasetFormat = DatasetFormat.AUTO,
    train_on_inputs: bool = False,
    num_proc: int | None = None,
):
    """Tokenize dataset with chat formatting and assistant-only loss masking."""
    fmt = dataset_format
    if fmt == DatasetFormat.AUTO and len(dataset) > 0:
        fmt = detect_format(dataset[0])

    logger.info("Dataset format: %s, samples: %d", fmt.value, len(dataset))
    mask_assistant_only = not train_on_inputs and fmt != DatasetFormat.TEXT

    def tokenize_batch(batch):
        rows = _rows_from_batch(batch)
        out: dict[str, list[Any]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for row in rows:
            if fmt == DatasetFormat.TEXT:
                encoded = _tokenize_text_row(row, tokenizer, max_seq_length)
            else:
                encoded = _tokenize_chat_row(
                    row,
                    fmt,
                    tokenizer,
                    max_seq_length,
                    mask_assistant_only=False,
                )
            out["input_ids"].append(encoded["input_ids"])
            out["attention_mask"].append(encoded["attention_mask"])
            out["labels"].append(encoded["labels"])
        return out

    def tokenize_masked_batch(batch):
        rows = _rows_from_batch(batch)
        out: dict[str, list[Any]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for row in rows:
            encoded = _tokenize_chat_row(
                row,
                fmt,
                tokenizer,
                max_seq_length,
                mask_assistant_only=True,
            )
            out["input_ids"].append(encoded["input_ids"])
            out["attention_mask"].append(encoded["attention_mask"])
            out["labels"].append(encoded["labels"])
        return out

    map_kwargs: dict[str, Any] = {"remove_columns": dataset.column_names}
    if num_proc and num_proc > 1:
        map_kwargs["num_proc"] = num_proc
    if mask_assistant_only:
        tokenized = dataset.map(tokenize_masked_batch, batched=True, **map_kwargs)
    else:
        tokenized = dataset.map(tokenize_batch, batched=True, **map_kwargs)
    return tokenized, fmt


def format_dataset_text(
    dataset,
    tokenizer,
    dataset_format: DatasetFormat = DatasetFormat.AUTO,
    *,
    num_proc: int | None = None,
):
    """Add `text` column for TRL SFTTrainer."""
    fmt = dataset_format
    if fmt == DatasetFormat.AUTO and len(dataset) > 0:
        fmt = detect_format(dataset[0])

    eos = getattr(tokenizer, "eos_token", "") or ""

    def add_text_batch(batch):
        texts = []
        for row in _rows_from_batch(batch):
            text = format_sample(row, fmt, tokenizer)
            if eos and not text.endswith(eos):
                text += eos
            texts.append(text)
        return {"text": texts}

    map_kwargs: dict[str, Any] = {
        "batched": True,
        "remove_columns": dataset.column_names,
    }
    if num_proc and num_proc > 1:
        map_kwargs["num_proc"] = num_proc
    return dataset.map(add_text_batch, **map_kwargs), fmt
