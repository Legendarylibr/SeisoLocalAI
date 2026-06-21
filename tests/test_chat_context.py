from __future__ import annotations

from forge.services.chat_context import compute_chat_context_status


def test_compute_chat_context_status_reports_trim_and_fill():
    history = [
        {"role": "user", "content": "hello", "created_at": ""},
        {"role": "assistant", "content": "hi there", "created_at": ""},
    ]
    status = compute_chat_context_status(history, max_tokens=512, n_ctx=4096)
    assert status["message_count"] == 2
    assert status["messages_included"] == 2
    assert status["messages_omitted"] == 0
    assert status["n_ctx"] >= 2048
    assert 0 < status["fill_ratio"] < 1
    assert status["context_tokens_used"] <= status["context_tokens_limit"]


def test_compute_chat_context_status_marks_trimmed_history(monkeypatch):
    monkeypatch.setattr("forge.services.chat_context._context_char_budget", lambda: 500)
    long = "word " * 5000
    history = [
        {"role": "user", "content": long, "created_at": ""},
        {"role": "assistant", "content": "ok", "created_at": ""},
        {"role": "user", "content": "latest question", "created_at": ""},
    ]
    status = compute_chat_context_status(history, max_tokens=2048, n_ctx=8192)
    assert status["history_trimmed"] is True
    assert status["char_used"] < status["char_total"]
