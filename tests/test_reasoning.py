from __future__ import annotations

from forge.services.reasoning import ReasoningStreamParser, split_reasoning_text


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
