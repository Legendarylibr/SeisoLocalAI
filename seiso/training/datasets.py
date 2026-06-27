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
            from seiso.memory.protection import jsonl_load_safe

            if jsonl_load_safe(p):
                logger.info("Large JSONL — loading via datasets mmap (%s)", p)
                return load_dataset("json", data_files=str(p), split=split)
            rows = []
            with p.open() as f:
                for line in f:
                    rows.append(json.loads(line))
            return Dataset.from_list(rows)
        if p.suffix == ".json":
            data = json.loads(p.read_text())
            return Dataset.from_list(data if isinstance(data, list) else [data])
        if p.is_dir():
            return load_dataset(str(p), split=split, revision="main")  # nosec B615: local path, revision pinned for hub fallback
    return load_dataset(str(path), split=split, revision="main")  # nosec B615: revision pinned


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
        return DatasetFormat.SHAREGPT if "conversations" in sample else DatasetFormat.CHAT
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
        rows = [
            dict(zip(batch.keys(), values, strict=True))
            for values in zip(*batch.values(), strict=True)
        ]
        texts = [format_sample(row, fmt, tokenizer) for row in rows]
        encoded = tokenizer(texts, truncation=True, max_length=max_seq_length, padding=False)
        encoded["labels"] = [list(ids) for ids in encoded["input_ids"]]
        return encoded

    def tokenize(sample):
        if mask_assistant_only:
            messages = extract_messages(sample, fmt)
            if messages and messages[-1].get("role") == "assistant":
                full_text = format_messages_for_prompt(
                    messages, tokenizer, add_generation_prompt=False
                )
                prompt_messages = messages[:-1]
                prompt_text = format_messages_for_prompt(
                    prompt_messages, tokenizer, add_generation_prompt=True
                )
                full_ids = tokenizer(
                    full_text, truncation=True, max_length=max_seq_length, padding=False
                )
                prompt_ids = tokenizer(
                    prompt_text, truncation=True, max_length=max_seq_length, padding=False
                )
                labels = _build_labels(full_ids["input_ids"], len(prompt_ids["input_ids"]))
                return {
                    "input_ids": full_ids["input_ids"],
                    "attention_mask": full_ids["attention_mask"],
                    "labels": labels,
                }

        text = format_sample(sample, fmt, tokenizer)
        encoded = tokenizer(text, truncation=True, max_length=max_seq_length, padding=False)
        encoded["labels"] = list(encoded["input_ids"])
        return encoded

    map_kwargs: dict[str, Any] = {"remove_columns": dataset.column_names}
    if num_proc and num_proc > 1:
        map_kwargs["num_proc"] = num_proc
    if mask_assistant_only:
        tokenized = dataset.map(tokenize, **map_kwargs)
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
        rows = [
            dict(zip(batch.keys(), values, strict=True))
            for values in zip(*batch.values(), strict=True)
        ]
        texts = []
        for row in rows:
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
