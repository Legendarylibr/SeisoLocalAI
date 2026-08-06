"""Unified thinking budget — quality-first, model-agnostic."""

from __future__ import annotations


def test_classify_task_quality_buckets():
    from seiso.chat.thinking import classify_task

    assert classify_task([{"role": "user", "content": "hi"}]) == "simple"
    assert classify_task([{"role": "user", "content": "write a song about the moon"}]) == "creative"
    assert (
        classify_task(
            [
                {
                    "role": "user",
                    "content": "Prove that the square root of 2 is irrational step by step",
                }
            ]
        )
        == "complex"
    )
    assert (
        classify_task([{"role": "user", "content": "What is the capital of France?"}]) == "general"
    )


def test_auto_policy_disables_thinking_for_creative_and_simple(monkeypatch):
    from seiso.chat.thinking import resolve_thinking_policy

    monkeypatch.delenv("SEISO_THINK_MODE", raising=False)
    monkeypatch.delenv("SEISO_OLLAMA_THINK", raising=False)
    monkeypatch.delenv("SEISO_THINK_MAX_TOKENS", raising=False)

    creative = resolve_thinking_policy(
        content_max_tokens=768,
        messages=[{"role": "user", "content": "write a song with better rhyme"}],
        model_key="Qwen/Qwen3-4B",
    )
    assert creative.enabled is False
    assert creative.api_value is False
    assert creative.decode_max_tokens == 768

    simple = resolve_thinking_policy(
        content_max_tokens=768,
        messages=[{"role": "user", "content": "hey"}],
        model_key="Qwen/Qwen3-4B",
    )
    assert simple.enabled is False


def test_auto_policy_enables_thinking_for_complex_reasoning(monkeypatch):
    from seiso.chat.thinking import resolve_thinking_policy

    monkeypatch.delenv("SEISO_THINK_MODE", raising=False)
    monkeypatch.delenv("SEISO_OLLAMA_THINK", raising=False)
    monkeypatch.setenv("SEISO_THINK_MAX_TOKENS", "128")
    monkeypatch.setenv("SEISO_THINK_BUDGET_RATIO", "0.25")
    monkeypatch.setenv("SEISO_CONTENT_RESERVE_RATIO", "0.70")

    policy = resolve_thinking_policy(
        content_max_tokens=768,
        messages=[
            {
                "role": "user",
                "content": "Solve this step by step and prove the algorithm is correct",
            }
        ],
        model_key="meta-llama/Llama-3.2-3B-Instruct",
    )
    assert policy.enabled is True
    assert policy.think_max_tokens > 0
    assert policy.think_max_tokens <= 128
    # Content majority reserved: decode expands by at most think_max.
    assert policy.decode_max_tokens == 768 + policy.think_max_tokens
    assert policy.content_max_tokens == 768


def test_auto_policy_reasoning_model_general_chat(monkeypatch):
    from seiso.chat.thinking import resolve_thinking_policy

    monkeypatch.delenv("SEISO_THINK_MODE", raising=False)
    monkeypatch.delenv("SEISO_OLLAMA_THINK", raising=False)
    monkeypatch.setenv("SEISO_THINK_MAX_TOKENS", "96")

    policy = resolve_thinking_policy(
        content_max_tokens=512,
        messages=[{"role": "user", "content": "Summarize the main idea of recursion."}],
        model_key="Qwen/Qwen3-4B-Q5_0.gguf",
    )
    # Reasoning-prone + non-trivial prompt → brief thinking allowed.
    assert policy.enabled is True
    assert policy.api_value in {True, "low", "medium"}
    assert 0 < policy.think_max_tokens <= 96


def test_content_reserve_limits_thinking(monkeypatch):
    from seiso.chat.thinking import thinking_max_tokens

    monkeypatch.setenv("SEISO_THINK_MAX_TOKENS", "256")
    monkeypatch.setenv("SEISO_THINK_BUDGET_RATIO", "0.5")
    monkeypatch.setenv("SEISO_CONTENT_RESERVE_RATIO", "0.70")
    # 30% of 200 = 60 max from reserve; ratio would want 100 → capped to 60.
    assert thinking_max_tokens(200, task="general") == 60


def test_stream_guard_caps_inline_think_tags():
    from seiso.chat.thinking import ThinkingStreamGuard

    guard = ThinkingStreamGuard(think_max_tokens=4)
    # Open think with enough body to exceed cap.
    emit, abort = guard.feed_text("<think>" + ("plan " * 20))
    assert emit == ""
    assert abort is True
    assert guard.capped is True


def test_stream_guard_passes_visible_content():
    from seiso.chat.thinking import ThinkingStreamGuard

    guard = ThinkingStreamGuard(think_max_tokens=8)
    emit, abort = guard.feed_text("Hello there, friend.")
    assert "Hello" in emit
    assert abort is False
    assert guard.saw_visible_content is True


def test_apply_thinking_policy_idempotent(monkeypatch):
    from seiso.chat.thinking import apply_thinking_policy

    monkeypatch.delenv("SEISO_THINK_MODE", raising=False)
    monkeypatch.delenv("SEISO_OLLAMA_THINK", raising=False)
    first = apply_thinking_policy(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 256,
            "model_name": "Qwen3-4B",
        }
    )
    second = apply_thinking_policy(first)
    assert first["think"] == second["think"]
    assert first.get("_thinking_policy_applied") is True
