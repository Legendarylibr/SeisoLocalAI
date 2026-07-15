"""Tests for length-stop detection and safe auto-continue helpers."""

from __future__ import annotations

from forge.services.generation_continue import (
    CONTINUE_USER_PROMPT,
    authoritative_pass_tokens,
    build_continue_messages,
    estimate_tokens_from_text,
    hit_length_limit,
    looks_long_form,
    max_auto_continues,
    next_pass_max_tokens,
    reply_still_truncated,
    resolve_auto_continue_limits,
    resolve_finish_reason,
    should_auto_continue,
    total_reply_token_budget,
)


def test_hit_length_limit_by_token_budget():
    assert hit_length_limit(2048, 2048) is True
    assert hit_length_limit(2047, 2048) is True
    assert hit_length_limit(100, 2048) is False


def test_hit_length_limit_by_finish_reason():
    assert hit_length_limit(12, 2048, finish_reason="length") is True
    assert hit_length_limit(12, 2048, finish_reason="stop") is False


def test_hit_length_limit_uses_text_estimate_when_meter_undercounts():
    # ~768 tokens of text but backend reported only 40.
    long_text = "word " * 800
    assert estimate_tokens_from_text(long_text) >= 500
    assert (
        hit_length_limit(
            40,
            512,
            finish_reason="stop",
            pass_text=long_text,
        )
        is True
    )


def test_reply_still_truncated_only_when_budget_exhausted():
    # Length-hit but multi-pass budget remains → do not show truncated banner.
    assert (
        reply_still_truncated(
            last_pass_tokens=512,
            pass_max_tokens=512,
            finish_reason="length",
            total_output_tokens=512,
            total_budget=32768,
            continues_used=0,
            max_continues=40,
        )
        is False
    )
    # Continues exhausted → truncated.
    assert (
        reply_still_truncated(
            last_pass_tokens=512,
            pass_max_tokens=512,
            finish_reason="length",
            total_output_tokens=512,
            total_budget=32768,
            continues_used=40,
            max_continues=40,
        )
        is True
    )
    # Total budget exhausted (less than one more 8-token chunk) → truncated.
    assert (
        reply_still_truncated(
            last_pass_tokens=512,
            pass_max_tokens=512,
            finish_reason="length",
            total_output_tokens=32761,
            total_budget=32768,
            continues_used=5,
            max_continues=40,
        )
        is True
    )
    # Natural stop → not truncated.
    assert (
        reply_still_truncated(
            last_pass_tokens=40,
            pass_max_tokens=512,
            finish_reason="stop",
            total_output_tokens=40,
            total_budget=32768,
            continues_used=0,
            max_continues=40,
        )
        is False
    )


def test_next_pass_max_tokens_respects_remaining_budget():
    assert (
        next_pass_max_tokens(
            base_pass_max_tokens=768,
            total_output_tokens=32000,
            total_budget=32768,
        )
        == 768
    )
    assert (
        next_pass_max_tokens(
            base_pass_max_tokens=768,
            total_output_tokens=32700,
            total_budget=32768,
        )
        == 68
    )


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


def test_build_continue_messages_trims_to_fixed_n_ctx():
    """Growing assistant text must fit the pinned window (no n_ctx growth)."""
    base = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "write a long essay"},
    ]
    huge = "word " * 8000
    msgs = build_continue_messages(base, huge, n_ctx=2048, max_tokens=512)
    # Still ends with continue cue; total content is bounded by the fixed window.
    assert msgs[-1]["content"] == CONTINUE_USER_PROMPT
    total_chars = sum(len(str(m.get("content") or "")) for m in msgs)
    # 2048 ctx − 512 gen leaves ~1500 tokens ≈ ~4800 chars; allow generous headroom.
    assert total_chars < len(huge)


def test_linear_decay_keeps_recent_assistant_tail():
    from forge.services.generation_continue import (
        decay_assistant_draft,
        linear_decay_fill_ratio,
        pack_continue_messages_linear_decay,
    )

    # More free headroom as the draft grows relative to n_ctx.
    assert linear_decay_fill_ratio(assistant_tokens=100, n_ctx=8192) > linear_decay_fill_ratio(
        assistant_tokens=6000, n_ctx=8192
    )

    head = "TITLE AND INTRO " * 40
    middle = "MIDDLE SECTION " * 400
    tail = "RECENT CLAUSE where generation stopped uniquely."
    draft = head + middle + tail
    shrunk = decay_assistant_draft(draft, token_budget=200)
    assert "RECENT CLAUSE where generation stopped uniquely." in shrunk
    assert len(shrunk) < len(draft)
    assert "omitted" in shrunk or shrunk.endswith("uniquely.")

    base = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "write a long research paper on fusion energy"},
        {"role": "assistant", "content": draft},
        {"role": "user", "content": CONTINUE_USER_PROMPT},
    ]
    packed = pack_continue_messages_linear_decay(base, n_ctx=2048, max_tokens=512)
    assert packed[-1]["content"] == CONTINUE_USER_PROMPT
    # Recent tail of the in-progress reply must survive packing.
    assistant_msgs = [m for m in packed if m["role"] == "assistant"]
    assert assistant_msgs
    assert "RECENT CLAUSE where generation stopped uniquely." in assistant_msgs[-1]["content"]
    # Never pack near the full hard budget (linear free headroom).
    total_chars = sum(len(str(m.get("content") or "")) for m in packed)
    assert total_chars < len(draft)


def test_max_auto_continues_explicit_and_clamped(monkeypatch):
    monkeypatch.setenv("SEISO_CHAT_AUTO_CONTINUE_MAX", "99")
    assert max_auto_continues() == 99
    monkeypatch.setenv("SEISO_CHAT_AUTO_CONTINUE_MAX", "999")
    assert max_auto_continues() == 256
    monkeypatch.setenv("SEISO_CHAT_AUTO_CONTINUE_MAX", "0")
    assert max_auto_continues() == 0


def test_max_auto_continues_derives_from_budget_and_pass_size(monkeypatch):
    """Native 512-token passes must not hard-stop long replies at 8 continues."""
    monkeypatch.delenv("SEISO_CHAT_AUTO_CONTINUE_MAX", raising=False)
    monkeypatch.setenv("SEISO_CHAT_AUTO_CONTINUE_TOTAL_TOKENS", "32768")
    # 32768 / 512 = 64 passes → 63 continues.
    assert max_auto_continues(pass_max_tokens=512) == 63
    # Larger per-pass needs fewer continues.
    assert max_auto_continues(pass_max_tokens=2048) == 15


def test_total_reply_token_budget_default_and_clamp(monkeypatch):
    monkeypatch.delenv("SEISO_CHAT_AUTO_CONTINUE_TOTAL_TOKENS", raising=False)
    assert total_reply_token_budget() == 32768
    monkeypatch.setenv("SEISO_CHAT_AUTO_CONTINUE_TOTAL_TOKENS", "999999")
    assert total_reply_token_budget() == 131072


def test_should_auto_continue_respects_total_budget():
    assert (
        should_auto_continue(
            pass_output_tokens=512,
            max_tokens=512,
            pass_text="partial answer that was cut",
            continues_used=0,
            total_output_tokens=8190,
            total_budget=8192,
        )
        is False
    )
    assert (
        should_auto_continue(
            pass_output_tokens=512,
            max_tokens=512,
            pass_text="partial answer that was cut",
            continues_used=0,
            total_output_tokens=1000,
            total_budget=8192,
        )
        is True
    )


def test_should_auto_continue_allows_many_passes_within_budget(monkeypatch):
    """After 8 continues (old hard cap), still continue if budget remains."""
    monkeypatch.delenv("SEISO_CHAT_AUTO_CONTINUE_MAX", raising=False)
    monkeypatch.setenv("SEISO_CHAT_AUTO_CONTINUE_TOTAL_TOKENS", "32768")
    assert (
        should_auto_continue(
            pass_output_tokens=512,
            max_tokens=512,
            pass_text="partial answer still going",
            continues_used=8,
            finish_reason="length",
            total_output_tokens=9 * 512,
            total_budget=32768,
        )
        is True
    )


def test_authoritative_pass_tokens_prefers_eval_count():
    assert authoritative_pass_tokens(400, {"eval_count": 512}) == 512
    assert authoritative_pass_tokens(600, {"eval_count": 512}) == 600
    assert authoritative_pass_tokens(100, None) == 100


def test_resolve_finish_reason():
    assert resolve_finish_reason(hit_length=True) == "length"
    assert resolve_finish_reason(hit_length=False) == "stop"
    assert resolve_finish_reason(hit_length=False, cancelled=True) == "cancelled"
    assert resolve_finish_reason(hit_length=False, explicit="length") == "length"


def test_looks_long_form_detects_paper_and_song():
    assert looks_long_form([{"role": "user", "content": "Write a research paper on fusion"}])
    assert looks_long_form([{"role": "user", "content": "Compose a full song with lyrics"}])
    assert not looks_long_form([{"role": "user", "content": "What is 2+2?"}])


def test_resolve_limits_scales_with_request_and_pass_size(monkeypatch):
    monkeypatch.delenv("SEISO_CHAT_AUTO_CONTINUE_MAX", raising=False)
    monkeypatch.delenv("SEISO_CHAT_AUTO_CONTINUE_TOTAL_TOKENS", raising=False)
    max_cont, total = resolve_auto_continue_limits(
        requested_max_tokens=48000,
        pass_max_tokens=768,
        messages=[{"role": "user", "content": "hi"}],
        headroom_mb=20000,
    )
    assert total >= 48000
    assert max_cont >= (total // 768) - 1
    assert max_cont <= 256


def test_resolve_limits_long_form_boosts_total(monkeypatch):
    monkeypatch.delenv("SEISO_CHAT_AUTO_CONTINUE_MAX", raising=False)
    monkeypatch.delenv("SEISO_CHAT_AUTO_CONTINUE_TOTAL_TOKENS", raising=False)
    _max_cont, total = resolve_auto_continue_limits(
        requested_max_tokens=2048,
        pass_max_tokens=512,
        messages=[{"role": "user", "content": "Write a detailed research paper on RLHF"}],
        headroom_mb=20000,
    )
    assert total >= 65536


def test_resolve_limits_low_headroom_caps_total(monkeypatch):
    monkeypatch.delenv("SEISO_CHAT_AUTO_CONTINUE_MAX", raising=False)
    monkeypatch.delenv("SEISO_CHAT_AUTO_CONTINUE_TOTAL_TOKENS", raising=False)
    _max_cont, total = resolve_auto_continue_limits(
        requested_max_tokens=2048,
        pass_max_tokens=512,
        messages=[{"role": "user", "content": "Write a research paper on RLHF"}],
        headroom_mb=1000,
    )
    assert total <= 4096


def test_resolve_limits_env_override_wins(monkeypatch):
    monkeypatch.setenv("SEISO_CHAT_AUTO_CONTINUE_MAX", "3")
    monkeypatch.setenv("SEISO_CHAT_AUTO_CONTINUE_TOTAL_TOKENS", "4096")
    max_cont, total = resolve_auto_continue_limits(
        requested_max_tokens=30000,
        pass_max_tokens=512,
        messages=[{"role": "user", "content": "Write a research paper"}],
        headroom_mb=30000,
    )
    assert max_cont == 3
    assert total == 4096
