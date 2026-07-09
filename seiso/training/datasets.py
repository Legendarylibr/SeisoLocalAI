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


def _chat_template_token_ids(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    max_length: int,
) -> list[int] | None:
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
    return ids[:max_length]


def _tokenize_with_assistant_mask(
    rows: list[dict[str, Any]],
    fmt: DatasetFormat,
    tokenizer,
    max_seq_length: int,
) -> dict[str, list[Any]] | None:
    """Single chat-template pass with assistant token masks when the tokenizer supports it."""
    input_ids_batch: list[list[int]] = []
    attention_batch: list[list[int]] = []
    labels_batch: list[list[int]] = []
    apply = tokenizer.apply_chat_template

    for sample in rows:
        messages = extract_messages(sample, fmt)
        if not messages or messages[-1].get("role") != "assistant":
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
        ids = list(ids)[:max_seq_length]
        mask = list(mask)[: len(ids)]
        if len(mask) < len(ids):
            mask = mask + [0] * (len(ids) - len(mask))
        labels = [tok if m else -100 for tok, m in zip(ids, mask, strict=False)]
        # Truncation can leave only prompt tokens; fall back if no assistant labels remain.
        if all(label == -100 for label in labels):
            return None
        input_ids_batch.append(ids)
        attention_batch.append([1] * len(ids))
        labels_batch.append(labels)

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attention_batch,
        "labels": labels_batch,
    }


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
        texts = [format_sample(row, fmt, tokenizer) for row in rows]
        encoded = tokenizer(
            texts, truncation=True, max_length=max_seq_length, padding=False
        )
        encoded["labels"] = [list(ids) for ids in encoded["input_ids"]]
        return encoded

    def tokenize_masked_batch(batch):
        rows = _rows_from_batch(batch)
        # Prefer a single apply_chat_template pass with assistant masks when supported.
        if _tokenizer_supports_assistant_mask(tokenizer):
            encoded = _tokenize_with_assistant_mask(
                rows, fmt, tokenizer, max_seq_length
            )
            if encoded is not None:
                return encoded

        full_texts: list[str] = []
        prompt_message_lists: list[list[dict[str, Any]] | None] = []

        for sample in rows:
            messages = extract_messages(sample, fmt)
            if messages and messages[-1].get("role") == "assistant":
                full_texts.append(
                    format_messages_for_prompt(
                        messages, tokenizer, add_generation_prompt=False
                    )
                )
                prompt_message_lists.append(messages[:-1])
            else:
                full_texts.append(format_sample(sample, fmt, tokenizer))
                prompt_message_lists.append(None)

        full = tokenizer(
            full_texts, truncation=True, max_length=max_seq_length, padding=False
        )
        # Prompt lengths via tokenize=True chat template (avoids string→re-encode).
        prompt_lengths: dict[int, int] = {}
        for idx, prompt_messages in enumerate(prompt_message_lists):
            if prompt_messages is None:
                continue
            prompt_ids = _chat_template_token_ids(
                tokenizer,
                prompt_messages,
                add_generation_prompt=True,
                max_length=max_seq_length,
            )
            if prompt_ids is not None:
                prompt_lengths[idx] = len(prompt_ids)
            else:
                prompt_text = format_messages_for_prompt(
                    prompt_messages, tokenizer, add_generation_prompt=True
                )
                prompt_lengths[idx] = len(
                    tokenizer(
                        prompt_text,
                        truncation=True,
                        max_length=max_seq_length,
                        padding=False,
                    )["input_ids"]
                )
        full["labels"] = [
            _build_labels(ids, prompt_lengths.get(idx, 0))
            for idx, ids in enumerate(full["input_ids"])
        ]
        return full

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
