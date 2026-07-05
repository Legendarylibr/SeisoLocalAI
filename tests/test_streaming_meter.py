from __future__ import annotations

import pytest

from seiso.inference.streaming import (
    StreamToken,
    StreamUpdate,
    merge_stream_updates,
)


def test_stream_token_normalizes_non_positive_counts():
    token = StreamToken("hi", 0)
    assert token.new_tokens == 1


def test_merge_stream_updates_single_passthrough():
    update = StreamUpdate(text="hi", output_tokens=2)
    assert merge_stream_updates([update]) is update


def test_merge_stream_updates_concatenates_and_keeps_latest_count():
    merged = merge_stream_updates(
        [
            StreamUpdate(text="Hel", output_tokens=1),
            StreamUpdate(text="lo ", output_tokens=2),
            StreamUpdate(text="world", output_tokens=3),
        ]
    )
    assert merged.text == "Hello world"
    assert merged.output_tokens == 3


def test_stream_producer_batch_chars_skips_rebatch_for_llama(monkeypatch):
    from seiso.inference.runner import _stream_batch_chars, _stream_producer_batch_chars

    monkeypatch.setenv("SEISO_STREAM_BATCH_CHARS", "64")
    assert _stream_batch_chars() == 64
    assert _stream_producer_batch_chars("llama") == 1
    assert _stream_producer_batch_chars("mlx") == 1
    assert _stream_producer_batch_chars("torch") == 64


def test_decode_batcher_flushes_first_token_immediately():
    from seiso.inference.runner import _DecodeBatcher

    batcher = _DecodeBatcher(batch_chars=64)
    first = batcher.push("Hi")
    assert first is not None
    assert first.text == "Hi"
    assert first.new_tokens == 1
    second = batcher.push(" there")
    assert second is None
    third = batcher.push("!" * 70)
    assert third is not None
    assert third.new_tokens >= 2


def test_merge_stream_updates_requires_input():
    import pytest

    with pytest.raises(ValueError, match="at least one"):
        merge_stream_updates([])


@pytest.mark.asyncio
async def test_runner_stream_updates_counts_decode_steps(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner
    from seiso.inference.streaming import StreamToken

    runner = LocalInferenceRunner()

    def _fake_iter(_payload, _model_path, _route, should_stop):
        assert not should_stop()
        yield StreamToken("a")
        yield StreamToken("bc", 2)

    monkeypatch.setattr(
        runner, "_resolve_route", lambda _payload, _path: ("llama", "/tmp/model.gguf")
    )

    async def _noop_switch(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(runner, "_ensure_model_switch", _noop_switch)
    monkeypatch.setattr(runner, "_iter_tokens", _fake_iter)
    monkeypatch.setattr(runner._pool, "bump_generation", lambda: 1)
    monkeypatch.setattr(runner._pool, "is_generation_active", lambda _gen: True)

    updates = [
        update
        async for update in runner.stream_updates(
            {
                "model_path": "/tmp/model.gguf",
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
    ]

    # Adjacent updates may be coalesced into one SSE frame for throughput, so
    # assert on concatenated text and the cumulative token count instead of
    # exact chunk boundaries.
    assert "".join(part.text for part in updates) == "abc"
    assert updates[-1].output_tokens == 3
