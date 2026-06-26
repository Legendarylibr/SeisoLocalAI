from __future__ import annotations

import pytest

from seiso.inference.streaming import StreamToken


def test_stream_token_normalizes_non_positive_counts():
    token = StreamToken("hi", 0)
    assert token.new_tokens == 1


@pytest.mark.asyncio
async def test_runner_stream_updates_counts_decode_steps(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner
    from seiso.inference.streaming import StreamToken

    runner = LocalInferenceRunner()

    def _fake_iter(_payload, _model_path, _route, should_stop):
        assert not should_stop()
        yield StreamToken("a")
        yield StreamToken("bc", 2)

    monkeypatch.setattr(runner, "_resolve_route", lambda _payload, _path: ("llama", "/tmp/model.gguf"))
    async def _noop_switch(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(runner, "_ensure_model_switch", _noop_switch)
    monkeypatch.setattr(runner, "_iter_tokens", _fake_iter)
    monkeypatch.setattr(runner._pool, "bump_generation", lambda: 1)
    monkeypatch.setattr(runner._pool, "is_generation_active", lambda _gen: True)

    updates = [
        update
        async for update in runner.stream_updates(
            {"model_path": "/tmp/model.gguf", "messages": [{"role": "user", "content": "hi"}]}
        )
    ]

    assert [part.text for part in updates] == ["a", "bc"]
    assert updates[-1].output_tokens == 3