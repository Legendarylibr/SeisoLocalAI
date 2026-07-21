"""Tests for training dataset preprocessing."""

from __future__ import annotations

import pytest

from seiso.training.config import DatasetFormat
from seiso.training.datasets import detect_format
from seiso.training.preprocess import (
    compute_eval_split_size,
    normalize_sample,
    parse_human_assistant_dialog,
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

    def map(self, fn, remove_columns=None, **kwargs):
        out = []
        for row in self._rows:
            out.append(fn(row))
        list(out[0].keys()) if out else []
        return _Rows(out)

    def filter(self, fn, **kwargs):
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
    assert (
        normalize_sample({"instruction": "x", "output": ""}, DatasetFormat.ALPACA)
        is None
    )


def test_normalize_preserves_newlines_and_indentation():
    code = "def add(a, b):\n    return a + b\n"
    row = normalize_sample(
        {
            "messages": [
                {"role": "user", "content": "Write add\r\n"},
                {"role": "assistant", "content": code},
            ]
        },
        DatasetFormat.CHAT,
    )
    assert row is not None
    assert row["messages"][-1]["content"] == "def add(a, b):\n    return a + b"

    text_row = normalize_sample({"text": "line1\n    indented\n"}, DatasetFormat.TEXT)
    assert text_row == {"text": "line1\n    indented"}


def test_normalize_preserves_first_line_indent_on_multiline():
    """Overall strip() would eat leading indent on line 1 — must not."""
    indented = "    def add(a, b):\n        return a + b\n"
    row = normalize_sample({"text": indented}, DatasetFormat.TEXT)
    assert row == {"text": "    def add(a, b):\n        return a + b"}


def test_normalize_text_accepts_code_columns():
    indented = "    def add(a, b):\n        return a + b\n"
    for key in ("code", "raw_content", "file_content"):
        row = normalize_sample({key: indented}, DatasetFormat.TEXT)
        assert row == {"text": "    def add(a, b):\n        return a + b"}, key
    # Prefer text over code when both present.
    row = normalize_sample(
        {"text": "plain", "code": "should_not_win"},
        DatasetFormat.TEXT,
    )
    assert row == {"text": "plain"}


def test_normalize_trims_single_line_prose_only():
    row = normalize_sample(
        {
            "messages": [
                {"role": "user", "content": "  hi  "},
                {"role": "assistant", "content": "  hello  "},
            ]
        },
        DatasetFormat.CHAT,
    )
    assert row is not None
    assert row["messages"][0]["content"] == "hi"
    assert row["messages"][1]["content"] == "hello"


def test_parse_human_assistant_preserves_code_blocks():
    text = (
        "Human: Fix this\n\n"
        "Assistant:\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```"
    )
    messages = parse_human_assistant_dialog(text)
    assert messages[-1]["role"] == "assistant"
    assert "def add(a, b):\n    return a + b" in messages[-1]["content"]


def test_parse_human_assistant_preserves_indented_first_line():
    """Role delimiter must not consume the next line's indentation."""
    text = "Human: x\n\nAssistant:\n    def f():\n        return 1"
    messages = parse_human_assistant_dialog(text)
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "    def f():\n        return 1"


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
    cleaned, stats, fmt = preprocess_training_dataset(
        ds, dataset_format=DatasetFormat.ALPACA
    )
    assert fmt == DatasetFormat.ALPACA
    assert len(cleaned) == 1
    assert stats["removed_invalid"] == 1
    assert stats["removed_duplicate"] == 1


def test_compute_eval_split_size_caps_holdout():
    assert compute_eval_split_size(10000, 0.05, 128) == 128
    assert compute_eval_split_size(50, 0.5, 128) == 25
    assert compute_eval_split_size(5, 0.05, 128) == 0


def test_parse_human_assistant_dialog_multiturn():
    text = (
        "\n\nHuman: What is 2+2?\n\n"
        "Assistant: 4\n\n"
        "Human: And 3+3?\n\n"
        "Assistant: 6"
    )
    messages = parse_human_assistant_dialog(text)
    assert len(messages) == 4
    assert messages[0] == {"role": "user", "content": "What is 2+2?"}
    assert messages[-1] == {"role": "assistant", "content": "6"}


def test_detect_format_preference_pairs():
    sample = {
        "chosen": "Human: hi\n\nAssistant: hello",
        "rejected": "Human: hi\n\nAssistant: nope",
    }
    assert detect_format(sample) == DatasetFormat.PREFERENCE


def test_normalize_preference_uses_chosen_turns():
    row = normalize_sample(
        {
            "chosen": "\n\nHuman: Explain gravity\n\nAssistant: Gravity pulls masses together.",
            "rejected": "\n\nHuman: Explain gravity\n\nAssistant: Magic.",
        },
        DatasetFormat.PREFERENCE,
    )
    assert row
    assert row["messages"][0]["role"] == "user"
    assert "gravity" in row["messages"][0]["content"].lower()
    assert row["messages"][-1]["role"] == "assistant"


def test_preprocess_preference_refuses_without_opt_in():
    ds = _Rows(
        [
            {
                "chosen": "\n\nHuman: Hi\n\nAssistant: Hello there.",
                "rejected": "\n\nHuman: Hi\n\nAssistant: Go away.",
            },
        ]
    )
    with pytest.raises(ValueError, match="Preference datasets"):
        preprocess_training_dataset(ds, dataset_format=DatasetFormat.AUTO)


def test_preprocess_preference_resolves_to_chat_with_opt_in():
    ds = _Rows(
        [
            {
                "chosen": "\n\nHuman: Hi\n\nAssistant: Hello there.",
                "rejected": "\n\nHuman: Hi\n\nAssistant: Go away.",
            },
            {
                "chosen": "\n\nHuman: Bye\n\nAssistant: Goodbye.",
                "rejected": "\n\nHuman: Bye\n\nAssistant: Nope.",
            },
        ]
    )
    cleaned, stats, fmt = preprocess_training_dataset(
        ds,
        dataset_format=DatasetFormat.AUTO,
        preference_as_sft=True,
    )
    assert fmt == DatasetFormat.CHAT
    assert len(cleaned) == 2
    assert stats["kept"] == 2
    assert stats["preference_as_sft"] is True
    assert "messages" in cleaned[0]
