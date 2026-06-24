"""Tests for training dataset preprocessing."""

from __future__ import annotations

from seiso.training.config import DatasetFormat
from seiso.training.preprocess import (
    compute_eval_split_size,
    normalize_sample,
    preprocess_training_dataset,
)


class _Rows:
    column_names: list[str]

    def __init__(self, rows: list[dict]):
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]

    def map(self, fn, remove_columns=None):
        out = []
        for row in self._rows:
            out.append(fn(row))
        list(out[0].keys()) if out else []
        return _Rows(out)

    def filter(self, fn):
        kept = [row for row in self._rows if fn(row)]
        return _Rows(kept)

    def select(self, indices):
        return _Rows([self._rows[i] for i in indices])

    def remove_columns(self, names):
        stripped = []
        for row in self._rows:
            stripped.append({k: v for k, v in row.items() if k not in names})
        return _Rows(stripped)


def test_normalize_alpaca_strips_and_requires_output():
    row = normalize_sample(
        {"instruction": "  hi ", "input": "", "output": " hello "},
        DatasetFormat.ALPACA,
    )
    assert row == {"instruction": "hi", "output": "hello"}
    assert normalize_sample({"instruction": "x", "output": ""}, DatasetFormat.ALPACA) is None


def test_normalize_chat_requires_assistant():
    ok = normalize_sample(
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        },
        DatasetFormat.CHAT,
    )
    assert ok and ok["messages"][-1]["content"] == "answer"
    bad = normalize_sample(
        {"messages": [{"role": "user", "content": "only user"}]}, DatasetFormat.CHAT
    )
    assert bad is None


def test_preprocess_drops_invalid_and_duplicates():
    ds = _Rows(
        [
            {"instruction": "a", "output": "1"},
            {"instruction": "a", "output": "1"},
            {"instruction": "b", "output": ""},
        ]
    )
    cleaned, stats, fmt = preprocess_training_dataset(ds, dataset_format=DatasetFormat.ALPACA)
    assert fmt == DatasetFormat.ALPACA
    assert len(cleaned) == 1
    assert stats["removed_invalid"] == 1
    assert stats["removed_duplicate"] == 1


def test_compute_eval_split_size_caps_holdout():
    assert compute_eval_split_size(10000, 0.05, 128) == 128
    assert compute_eval_split_size(50, 0.5, 128) == 25
    assert compute_eval_split_size(5, 0.05, 128) == 0
