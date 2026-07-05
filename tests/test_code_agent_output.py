from __future__ import annotations

from forge.services.llm_output import sanitize_agent_output, strip_reasoning_artifacts


def test_strip_reasoning_artifacts_redacts_qwen_think_block():
    raw = "\x3cthink\x3eplan steps\x3c/think\x3e\nHere is the patch summary."
    assert strip_reasoning_artifacts(raw) == "Here is the patch summary."


def test_strip_reasoning_artifacts_redacts_redacted_thinking():
    raw = (
        "<think>internal notes</think>\n"
        "Applied a one-line fix in `api.ts`."
    )
    assert strip_reasoning_artifacts(raw) == "Applied a one-line fix in `api.ts`."


def test_strip_reasoning_artifacts_redacts_gemma_reasoning_section():
    raw = (
        "Reasoning: inspect imports first\n\n"
        "Answer: Added the missing export."
    )
    assert "Reasoning:" not in strip_reasoning_artifacts(raw)
    assert "Added the missing export." in strip_reasoning_artifacts(raw)


def test_sanitize_agent_output_removes_tool_markup_and_thinking():
    raw = (
        'hidden<tool_call>{"name":"read_file","arguments":{"path":"a.ts"}}'
        "</tool_call>Updated `a.ts`."
    )
    cleaned = sanitize_agent_output(raw)
    assert "<tool_call>" not in cleaned
    assert cleaned.endswith("Updated `a.ts`.")
