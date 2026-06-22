"""Tests for math dataset formatting."""

from __future__ import annotations

from seiso.models.chat_format import extract_messages
from seiso.training.config import DatasetFormat
from seiso.training.datasets import detect_format


def test_detect_format_metamathqa():
    sample = {
        "type": "MATH_AnsAug",
        "query": "What is 2+2?",
        "response": "4",
    }
    assert detect_format(sample) == DatasetFormat.ALPACA


def test_extract_messages_metamathqa():
    sample = {"query": "What is 2+2?", "response": "4"}
    messages = extract_messages(sample, DatasetFormat.ALPACA)
    assert messages == [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]


def test_extract_messages_gsm8k():
    sample = {"question": "Jan has 3 apples.", "answer": "3"}
    messages = extract_messages(sample, DatasetFormat.ALPACA)
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
