from __future__ import annotations

from forge.api.schemas.inference import ChatRequest
from forge.services.reasoning import ReasoningStreamParser, split_reasoning_text
from seiso.inference.llamaswap import OllamaClient, ollama_thinking_enabled
from seiso.models.chat_format import format_messages_for_prompt


def test_split_reasoning_text_keeps_answer_separate():
    answer, reasoning = split_reasoning_text(
        "<think>Check the inputs.\nThen calculate.</think>\nThe answer is 42."
    )

    assert reasoning == "Check the inputs.\nThen calculate."
    assert answer == "The answer is 42."


def test_reasoning_stream_parser_handles_tags_split_across_chunks():
    parser = ReasoningStreamParser()
    output: list[tuple[str, str]] = []

    for chunk in ("<thi", "nk>step one", "\nstep two</th", "ink>final"):
        output.extend(parser.feed(chunk))
    output.extend(parser.finish())

    assert "".join(value for kind, value in output if kind == "reasoning") == (
        "step one\nstep two"
    )
    assert "".join(value for kind, value in output if kind == "answer") == "final"


def test_reasoning_stream_parser_preserves_plain_answers():
    parser = ReasoningStreamParser()
    output = [*parser.feed("Nothing special here."), *parser.finish()]

    assert output == [("answer", "Nothing special here.")]


def test_chat_requests_enable_reasoning_by_default():
    assert ChatRequest().reasoning is True
    assert ChatRequest(reasoning=False).reasoning is False


def test_chat_template_enables_native_thinking_with_legacy_fallback():
    calls: list[dict] = []

    class ThinkingTokenizer:
        def apply_chat_template(self, _messages, **kwargs):
            calls.append(kwargs)
            return "thinking prompt"

    rendered = format_messages_for_prompt(
        [{"role": "user", "content": "solve this"}],
        ThinkingTokenizer(),
        enable_thinking=True,
    )

    assert rendered == "thinking prompt"
    assert calls[-1]["enable_thinking"] is True

    class LegacyTokenizer:
        def apply_chat_template(
            self,
            _messages,
            *,
            tokenize=False,
            add_generation_prompt=True,
        ):
            return f"{tokenize}:{add_generation_prompt}"

    assert (
        format_messages_for_prompt(
            [{"role": "user", "content": "hello"}],
            LegacyTokenizer(),
            enable_thinking=True,
        )
        == "False:True"
    )


def test_ollama_thinking_flag_is_limited_to_supported_models():
    assert ollama_thinking_enabled({"reasoning": True}, "/models/Qwen3-8B.gguf")
    assert ollama_thinking_enabled(
        {"reasoning": True, "model_metadata": {"repo_id": "deepseek-ai/DeepSeek-R1"}},
        "/models/model.gguf",
    )
    assert not ollama_thinking_enabled({"reasoning": True}, "/models/Llama-3.2.gguf")
    assert not ollama_thinking_enabled({"reasoning": False}, "/models/Qwen3-8B.gguf")


def test_ollama_request_activates_supported_thinking_mode(monkeypatch):
    client = OllamaClient()
    monkeypatch.setattr(
        "seiso.inference.llamaswap.plan_sidecar_request",
        lambda _payload, _path: ([{"role": "user", "content": "solve"}], 2048, 512),
    )
    monkeypatch.setattr(client, "_load_options", lambda _path, **_kwargs: {"num_ctx": 2048})
    monkeypatch.setattr(client, "_resolve_model", lambda _path, _payload: "qwen3")
    monkeypatch.setattr(
        "seiso.inference.llamaswap.sidecar_ollama_keep_alive",
        lambda **_kwargs: None,
    )

    body = client._request_body(
        {"reasoning": True},
        "/models/Qwen3-8B.gguf",
        stream=True,
    )

    assert body["think"] is True
