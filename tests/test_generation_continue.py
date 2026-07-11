"""Tests for length-stop detection and safe auto-continue helpers."""

from __future__ import annotations

from forge.services.generation_continue import (
    CONTINUE_USER_PROMPT,
    build_continue_messages,
    hit_length_limit,
    max_auto_continues,
    resolve_finish_reason,
    should_auto_continue,
)


def test_hit_length_limit_by_token_budget():
    assert hit_length_limit(2048, 2048) is True
    assert hit_length_limit(2047, 2048) is True
    assert hit_length_limit(100, 2048) is False


def test_hit_length_limit_by_finish_reason():
    assert hit_length_limit(12, 2048, finish_reason="length") is True
    assert hit_length_limit(12, 2048, finish_reason="stop") is False


def test_should_auto_continue_respects_cap_and_cancel(monkeypatch):
    monkeypatch.setenv("SEISO_CHAT_AUTO_CONTINUE_MAX", "2")
    assert (
        should_auto_continue(
            pass_output_tokens=512,
            max_tokens=512,
            pass_text="partial answer that was cut",
            continues_used=0,
        )
        is True
    )
    assert (
        should_auto_continue(
            pass_output_tokens=512,
            max_tokens=512,
            pass_text="partial",
            continues_used=2,
        )
        is False
    )
    assert (
        should_auto_continue(
            pass_output_tokens=512,
            max_tokens=512,
            pass_text="partial",
            continues_used=0,
            cancelled=True,
        )
        is False
    )


def test_should_not_continue_on_natural_stop():
    assert (
        should_auto_continue(
            pass_output_tokens=40,
            max_tokens=2048,
            pass_text="Complete answer.",
            continues_used=0,
            finish_reason="stop",
        )
        is False
    )


def test_build_continue_messages_appends_partial_and_cue():
    base = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "write a long essay"},
    ]
    msgs = build_continue_messages(base, "Once upon a time")
    assert msgs[0]["role"] == "system"
    assert msgs[-2] == {"role": "assistant", "content": "Once upon a time"}
    assert msgs[-1] == {"role": "user", "content": CONTINUE_USER_PROMPT}


def test_max_auto_continues_clamped(monkeypatch):
    monkeypatch.setenv("SEISO_CHAT_AUTO_CONTINUE_MAX", "99")
    assert max_auto_continues() == 4
    monkeypatch.setenv("SEISO_CHAT_AUTO_CONTINUE_MAX", "0")
    assert max_auto_continues() == 0


def test_resolve_finish_reason():
    assert resolve_finish_reason(hit_length=True) == "length"
    assert resolve_finish_reason(hit_length=False) == "stop"
    assert resolve_finish_reason(hit_length=False, cancelled=True) == "cancelled"
    assert resolve_finish_reason(hit_length=False, explicit="length") == "length"
