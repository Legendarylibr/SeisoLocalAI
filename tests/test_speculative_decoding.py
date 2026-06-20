"""Tests for speculative decoding."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from seiso.inference.speculative import (
    TorchSpeculativeBundle,
    default_num_speculative_tokens,
    iter_speculative_tokens,
)


class _FakeTokenizer:
    def __init__(self) -> None:
        self._tokens = [1, 2, 3]

    def __call__(self, prompt: str, return_tensors: str = "pt"):
        import torch

        return SimpleNamespace(input_ids=torch.tensor([self._tokens], dtype=torch.long))

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return " ".join(str(int(t)) for t in token_ids.tolist())


class _FakeModel:
    def __init__(self, next_id: int) -> None:
        import torch

        self.next_id = next_id
        self.device = torch.device("cpu")

    def parameters(self):
        import torch

        yield torch.zeros(1)

    def __call__(self, input_ids):
        import torch

        batch, length = input_ids.shape
        vocab = 16
        logits = torch.zeros(batch, length, vocab)
        logits[:, :, self.next_id] = 10.0
        return SimpleNamespace(logits=logits)


def test_default_num_speculative_tokens_from_payload():
    assert default_num_speculative_tokens({"num_speculative_tokens": 3}) == 3


def test_default_num_speculative_tokens_env_fallback(monkeypatch):
    monkeypatch.setenv("SEISO_SPECULATIVE_TOKENS", "7")
    assert default_num_speculative_tokens({}) == 7


def test_iter_speculative_tokens_streams_matching_draft_and_target():
    tok = _FakeTokenizer()
    bundle = TorchSpeculativeBundle(
        target_model=_FakeModel(4),
        target_tokenizer=tok,
        draft_model=_FakeModel(4),
        draft_tokenizer=tok,
    )

    chunks = list(
        iter_speculative_tokens(
            bundle=bundle,
            prompt="hello",
            max_new_tokens=3,
            num_speculative_tokens=2,
        )
    )

    assert chunks
    assert any("4" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_runner_routes_to_speculative_stream(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    seen: dict[str, str] = {}

    async def _noop_switch(_path: str, *, draft_path: str | None = None) -> None:
        seen["draft_path"] = draft_path or ""

    def _fake_speculative(_payload, _model_path, should_stop):
        assert not should_stop()
        yield "spec"

    monkeypatch.setattr(runner, "_ensure_model_switch", _noop_switch)
    monkeypatch.setattr(runner, "_torch_speculative_stream", _fake_speculative)
    monkeypatch.setattr(runner._pool, "bump_generation", lambda: 1)
    monkeypatch.setattr(runner._pool, "is_generation_active", lambda _gen: True)

    tokens = [
        token
        async for token in runner.stream(
            {
                "model_path": "/tmp/target",
                "draft_model_path": "/tmp/draft",
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
    ]

    assert tokens == ["spec"]
    assert seen["draft_path"] == "/tmp/draft"


def test_model_pool_status_includes_draft_path():
    from seiso.inference.model_pool import BackendKind, LoadedModel, ModelPool

    pool = ModelPool()
    pool._active = LoadedModel(
        key="spec:/tmp/target:/tmp/draft",
        backend=BackendKind.TORCH,
        handle=MagicMock(),
        meta={"path": "/tmp/target", "draft_path": "/tmp/draft"},
    )
    status = pool.status()
    assert status["draft_path"] == "/tmp/draft"
