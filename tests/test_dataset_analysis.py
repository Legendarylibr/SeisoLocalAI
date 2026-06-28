"""Tests for research-grade dataset analysis."""

from __future__ import annotations

from seiso.training.config import DatasetFormat
from seiso.training.dataset_analysis import (
    _infer_domain,
    _length_stats,
    build_dataset_training_config,
    detect_format_consensus,
)
from seiso.training.preprocess import normalize_sample


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
        return _Rows([fn(row) for row in self._rows])

    def filter(self, fn):
        return _Rows([row for row in self._rows if fn(row)])

    def select(self, indices):
        return _Rows([self._rows[i] for i in indices])

    def remove_columns(self, names):
        return _Rows(
            [{k: v for k, v in row.items() if k not in names} for row in self._rows]
        )


def test_detect_format_consensus_majority_vote():
    samples = [
        {"instruction": "a", "output": "1"},
        {"instruction": "b", "output": "2"},
        {"messages": [{"role": "user", "content": "x"}]},
    ]
    fmt, confidence, meta = detect_format_consensus(samples)
    assert fmt == DatasetFormat.ALPACA
    assert confidence >= 0.66
    assert meta["votes"]["alpaca"] == 2


def test_infer_domain_instruction_vs_code():
    domain, label = _infer_domain(DatasetFormat.ALPACA, ["instruction", "output"])
    assert domain == "instruction_tuning"
    assert "Instruction" in label

    code_domain, _ = _infer_domain(DatasetFormat.TEXT, ["code", "language"])
    assert code_domain == "code_pretraining"


def test_normalize_prompt_completion_rows():
    row = normalize_sample(
        {"prompt": "Write hello world", "completion": "print('hello')"},
        DatasetFormat.ALPACA,
    )
    assert row == {"instruction": "Write hello world", "output": "print('hello')"}


def test_build_dataset_training_config_text_domain():
    cfg = build_dataset_training_config(
        resolved_format=DatasetFormat.TEXT,
        domain="causal_lm",
        kept=50_000,
        length_stats={"estimated_tokens_p95": 900},
    )
    assert cfg["dataset_format"] == "text"
    assert cfg["train_on_responses_only"] is False
    assert cfg["packing"] is True
    assert cfg["max_seq_length"] >= 512


def test_length_stats_percentiles():
    stats = _length_stats(
        [{"text": "a" * 10}, {"text": "b" * 100}, {"text": "c" * 1000}]
    )
    assert stats["chars_min"] == 10
    assert stats["chars_p50"] == 100
    assert stats["chars_max"] == 1000
