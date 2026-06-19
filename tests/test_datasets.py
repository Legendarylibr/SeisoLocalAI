"""Tests for dataset tokenization and label masking."""

from pathlib import Path

import pytest

from seiso.training.config import DatasetFormat
from seiso.training.datasets import prepare_tokenized_dataset


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            parts.append(f"{m['role']}: {m['content']}")
        if add_generation_prompt:
            parts.append("assistant:")
        text = "\n".join(parts)
        if tokenize:
            return [len(p) for p in parts]
        return text

    def __call__(self, text, truncation=True, max_length=2048, padding=False):
        ids = [hash(text) % 1000, len(text) % 100, 42]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def pad(self, features, padding=True, return_tensors="pt"):
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": []}
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [0] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
        return {k: torch.tensor(v) for k, v in batch.items()}


def test_cli_dataset_outside_data_dir_allowed(tmp_path: Path):
    """CLI training may reference repo-local datasets without sandbox_root."""
    from seiso.training.datasets import load_training_dataset

    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"text":"hello"}\n')
    ds = load_training_dataset(dataset, sandbox_root=None)
    assert len(ds) == 1


def test_sandbox_blocks_outside_path(tmp_path: Path):
    from seiso.security import SecurityError
    from seiso.training.datasets import load_training_dataset

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"text":"x"}\n')
    with pytest.raises(SecurityError):
        load_training_dataset(outside, sandbox_root=sandbox)


def test_chat_labels_mask_prompt():
    class _Rows:
        column_names = ["messages"]

        def __init__(self, rows):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        def __getitem__(self, idx):
            return self._rows[idx]

        def map(self, fn, remove_columns=None):
            out = []
            for row in self._rows:
                out.append(fn(row))
            return _Rows(out)

    ds = _Rows(
        [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}]
    )
    tok = _FakeTokenizer()
    tokenized, fmt = prepare_tokenized_dataset(
        ds, tok, max_seq_length=128, dataset_format=DatasetFormat.CHAT
    )
    assert fmt == DatasetFormat.CHAT
    labels = tokenized[0]["labels"]
    assert labels[0] == -100
