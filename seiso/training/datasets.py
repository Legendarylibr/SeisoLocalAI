"""Dataset loading and chat-template formatting for HF training."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from seiso.models.chat_format import extract_messages, format_messages_for_prompt
from seiso.training.config import DatasetFormat

logger = logging.getLogger(__name__)


def load_training_dataset(path: str | Path, split: str = "train", *, sandbox_root: Path | None = None):
    """Load dataset from HF hub ID, JSON/JSONL file, or directory."""
    from seiso.security import assert_within

    p = Path(path).expanduser()
    if p.exists() and sandbox_root is not None:
        assert_within(sandbox_root, p)

    from datasets import Dataset, load_dataset

    if p.exists():
        if p.suffix == ".jsonl":
            rows = []
            with p.open() as f:
                for line in f:
                    rows.append(json.loads(line))
            return Dataset.from_list(rows)
        if p.suffix == ".json":
            data = json.loads(p.read_text())
            return Dataset.from_list(data if isinstance(data, list) else [data])
        if p.is_dir():
            return load_dataset(str(p), split=split)
    return load_dataset(str(path), split=split)


def detect_format(sample: dict) -> DatasetFormat:
    if "conversations" in sample or "messages" in sample:
        return DatasetFormat.SHAREGPT if "conversations" in sample else DatasetFormat.CHAT
    if "instruction" in sample and "output" in sample:
        return DatasetFormat.ALPACA
    if "text" in sample or "content" in sample:
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
):
    """Tokenize dataset with chat formatting and assistant-only loss masking."""
    fmt = dataset_format
    if fmt == DatasetFormat.AUTO and len(dataset) > 0:
        fmt = detect_format(dataset[0])

    logger.info("Dataset format: %s, samples: %d", fmt.value, len(dataset))
    mask_assistant_only = not train_on_inputs and fmt != DatasetFormat.TEXT

    def tokenize(sample):
        if mask_assistant_only:
            messages = extract_messages(sample, fmt)
            if messages and messages[-1].get("role") == "assistant":
                full_text = format_messages_for_prompt(messages, tokenizer, add_generation_prompt=False)
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
                return {"input_ids": full_ids["input_ids"], "attention_mask": full_ids["attention_mask"], "labels": labels}

        text = format_sample(sample, fmt, tokenizer)
        encoded = tokenizer(text, truncation=True, max_length=max_seq_length, padding=False)
        encoded["labels"] = list(encoded["input_ids"])
        return encoded

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    return tokenized, fmt


def format_dataset_text(dataset, tokenizer, dataset_format: DatasetFormat = DatasetFormat.AUTO):
    """Add `text` column for TRL SFTTrainer."""
    fmt = dataset_format
    if fmt == DatasetFormat.AUTO and len(dataset) > 0:
        fmt = detect_format(dataset[0])

    eos = getattr(tokenizer, "eos_token", "") or ""

    def add_text(sample):
        text = format_sample(sample, fmt, tokenizer)
        if eos and not text.endswith(eos):
            text += eos
        return {"text": text}

    return dataset.map(add_text, remove_columns=dataset.column_names), fmt
