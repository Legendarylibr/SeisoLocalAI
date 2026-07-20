"""Tests for dataset tokenization and label masking."""

from __future__ import annotations

from pathlib import Path

import pytest

from seiso.training.config import DatasetFormat
from seiso.training.datasets import (
    format_dataset_text,
    prepare_tokenized_dataset,
    should_disable_packing_for_response_mask,
)


class _Rows:
    def __init__(self, rows, column_names=None):
        self._rows = rows
        self.column_names = column_names or list(rows[0].keys())
        self.map_kwargs = None

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]

    def map(self, fn, remove_columns=None, **kwargs):
        self.map_kwargs = kwargs
        assert kwargs.get("batched") is True
        batch = {key: [row[key] for row in self._rows] for key in self.column_names}
        mapped = fn(batch)
        out = [
            {key: value[i] for key, value in mapped.items()}
            for i in range(len(self._rows))
        ]
        return _Rows(out, column_names=list(mapped.keys()))


class _ChatTokenizer:
    """Deterministic chat-template tokenizer with optional assistant masks."""

    eos_token = "<eos>"
    eos_token_id = 99

    def __init__(self, *, support_assistant_mask: bool = True, fail_mask_on=None):
        self.support_assistant_mask = support_assistant_mask
        self.fail_mask_on = set(fail_mask_on or [])
        self._mask_calls = 0

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        return_dict=False,
        return_assistant_tokens_mask=False,
    ):
        if return_assistant_tokens_mask and not self.support_assistant_mask:
            raise TypeError("return_assistant_tokens_mask not supported")

        parts: list[tuple[str, list[int], bool]] = []
        # BOS
        ids: list[int] = [1]
        mask: list[int] = [0]
        for m in messages:
            role = m["role"]
            content = m["content"]
            role_ids = [10 if role == "user" else 20, len(content)]
            content_ids = [ord(c) % 50 + 30 for c in content] or [31]
            turn_ids = role_ids + content_ids
            is_asst = role == "assistant"
            ids.extend(turn_ids)
            mask.extend([1 if is_asst else 0] * len(turn_ids))
            parts.append((role, turn_ids, is_asst))

        if add_generation_prompt:
            ids.extend([20, 2])  # assistant header
            mask.extend([0, 0])

        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if add_generation_prompt:
            text += "\nassistant:"

        if not tokenize:
            return text

        if return_assistant_tokens_mask or return_dict:
            self._mask_calls += 1
            key = tuple((m["role"], m["content"]) for m in messages)
            if key in self.fail_mask_on:
                raise RuntimeError("forced mask failure")
            if return_dict or return_assistant_tokens_mask:
                return {
                    "input_ids": ids,
                    "assistant_masks": mask,
                }
        return ids

    def __call__(self, text, truncation=True, max_length=2048, padding=False):
        if isinstance(text, list):
            rows = [
                self(
                    item, truncation=truncation, max_length=max_length, padding=padding
                )
                for item in text
            ]
            return {
                "input_ids": [row["input_ids"] for row in rows],
                "attention_mask": [row["attention_mask"] for row in rows],
            }
        # Deterministic string encode distinct from chat-template ids.
        ids = [hash(text) % 1000, len(text) % 100, 42]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


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


def test_single_turn_masks_prompt_and_supervises_assistant_plus_eos():
    ds = _Rows(
        [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        ]
    )
    tok = _ChatTokenizer(support_assistant_mask=True)
    tokenized, fmt = prepare_tokenized_dataset(
        ds, tok, max_seq_length=128, dataset_format=DatasetFormat.CHAT
    )
    assert fmt == DatasetFormat.CHAT
    ids = tokenized[0]["input_ids"]
    labels = tokenized[0]["labels"]
    assert ids[-1] == tok.eos_token_id
    assert labels[-1] == tok.eos_token_id
    assert labels[0] == -100  # BOS masked
    assert any(label != -100 for label in labels[:-1])
    # User turn tokens must be masked.
    assert labels[1] == -100
    assert labels[2] == -100


def test_multi_turn_mask_api_supervises_both_assistant_spans():
    ds = _Rows(
        [
            {
                "messages": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                    {"role": "user", "content": "c"},
                    {"role": "assistant", "content": "d"},
                ]
            }
        ]
    )
    tok = _ChatTokenizer(support_assistant_mask=True)
    tokenized, _ = prepare_tokenized_dataset(
        ds, tok, max_seq_length=256, dataset_format=DatasetFormat.CHAT
    )
    labels = tokenized[0]["labels"]
    # Two assistant content tokens 'b' and 'd' (ord % 50 + 30) must be supervised.
    supervised = [lab for lab in labels if lab != -100]
    assert (ord("b") % 50 + 30) in supervised
    assert (ord("d") % 50 + 30) in supervised


def test_multi_turn_span_fallback_without_mask_api():
    ds = _Rows(
        [
            {
                "messages": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                    {"role": "user", "content": "c"},
                    {"role": "assistant", "content": "d"},
                ]
            }
        ]
    )
    tok = _ChatTokenizer(support_assistant_mask=False)
    tokenized, _ = prepare_tokenized_dataset(
        ds, tok, max_seq_length=256, dataset_format=DatasetFormat.CHAT
    )
    labels = tokenized[0]["labels"]
    supervised = [lab for lab in labels if lab != -100]
    assert (ord("b") % 50 + 30) in supervised
    assert (ord("d") % 50 + 30) in supervised
    assert labels[0] == -100


def test_keep_end_preserves_assistant_labels():
    long_user = "u" * 80
    ds = _Rows(
        [
            {
                "messages": [
                    {"role": "user", "content": long_user},
                    {"role": "assistant", "content": "OK"},
                ]
            }
        ]
    )
    tok = _ChatTokenizer(support_assistant_mask=True)
    # Full sequence is longer than 20; keep_end must retain assistant + eos.
    tokenized, _ = prepare_tokenized_dataset(
        ds, tok, max_seq_length=20, dataset_format=DatasetFormat.CHAT
    )
    labels = tokenized[0]["labels"]
    assert any(lab != -100 for lab in labels)
    assert tokenized[0]["input_ids"][-1] == tok.eos_token_id


def test_per_row_mask_failure_does_not_break_siblings():
    good = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]
    }
    bad_key = (
        ("user", "bad"),
        ("assistant", "nope"),
    )
    bad = {
        "messages": [
            {"role": "user", "content": "bad"},
            {"role": "assistant", "content": "nope"},
        ]
    }
    ds = _Rows([good, bad])
    tok = _ChatTokenizer(support_assistant_mask=True, fail_mask_on={bad_key})
    tokenized, _ = prepare_tokenized_dataset(
        ds, tok, max_seq_length=128, dataset_format=DatasetFormat.CHAT
    )
    good_labels = tokenized[0]["labels"]
    bad_labels = tokenized[1]["labels"]
    assert good_labels[0] == -100
    assert any(lab != -100 for lab in good_labels)
    # Bad row falls back to spans (same template encode) — still masked, not batch-wide fail.
    assert bad_labels[0] == -100
    assert any(lab != -100 for lab in bad_labels)


def test_text_tokenization_uses_batched_map():
    ds = _Rows([{"text": "hello"}, {"text": "world"}])
    tokenized, fmt = prepare_tokenized_dataset(
        ds,
        _ChatTokenizer(),
        max_seq_length=128,
        dataset_format=DatasetFormat.TEXT,
    )

    assert fmt == DatasetFormat.TEXT
    assert ds.map_kwargs == {"batched": True}
    assert tokenized[0]["labels"] == tokenized[0]["input_ids"]
    assert tokenized[0]["input_ids"][-1] == _ChatTokenizer.eos_token_id


def test_format_dataset_text_uses_batched_map_and_appends_eos():
    ds = _Rows([{"text": "hello"}, {"text": "already<eos>"}])
    formatted, fmt = format_dataset_text(
        ds, _ChatTokenizer(), DatasetFormat.TEXT, num_proc=2
    )

    assert fmt == DatasetFormat.TEXT
    assert ds.map_kwargs == {"batched": True, "num_proc": 2}
    assert formatted[0]["text"] == "hello<eos>"
    assert formatted[1]["text"] == "already<eos>"


@pytest.mark.parametrize(
    ("packing", "responses_only", "fmt", "expected"),
    [
        (True, True, DatasetFormat.CHAT, True),
        (True, True, DatasetFormat.ALPACA, True),
        (True, True, DatasetFormat.TEXT, False),
        (True, False, DatasetFormat.CHAT, False),
        (False, True, DatasetFormat.CHAT, False),
    ],
)
def test_should_disable_packing_for_response_mask(
    packing, responses_only, fmt, expected
):
    assert (
        should_disable_packing_for_response_mask(packing, responses_only, fmt)
        is expected
    )
