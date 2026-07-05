"""Tests for speculative decoding."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from seiso.inference.speculative import (
    DFlashDraftSpeculativeBundle,
    TorchSpeculativeBundle,
    _decode_new_text,
    default_num_speculative_tokens,
    iter_speculative_tokens,
    iter_speculative_tokens_dflash,
)


class _FakeTokenizer:
    def __init__(self) -> None:
        self._tokens = [1, 2, 3]

    def __call__(self, prompt: str, return_tensors: str = "pt"):
        import torch

        return SimpleNamespace(input_ids=torch.tensor([self._tokens], dtype=torch.long))

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        ids = token_ids if isinstance(token_ids, list) else token_ids.tolist()
        return " ".join(str(int(t)) for t in ids)


class _BpeLikeTokenizer(_FakeTokenizer):
    """Tokenizer where per-token decode differs from full-sequence decode."""

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        ids = token_ids if isinstance(token_ids, list) else token_ids.tolist()
        if ids == [10, 11]:
            return "AB"
        mapping = {10: "A", 11: "X", 12: "C", 1: "1", 2: "2", 3: "3", 4: "4"}
        return "".join(mapping.get(int(i), str(i)) for i in ids)


class _FakeModel:
    def __init__(self, next_id: int) -> None:
        import torch

        self.next_id = next_id
        self.device = torch.device("cpu")

    def parameters(self):
        import torch

        yield torch.zeros(1)

    def __call__(self, input_ids, past_key_values=None, use_cache=False):
        import torch

        batch, length = input_ids.shape
        vocab = 16
        logits = torch.zeros(batch, length, vocab)
        logits[:, :, self.next_id] = 10.0
        cached = (past_key_values or 0) + length if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=cached)


class _RejectingFakeModel:
    """Target accepts one proposed 5, then rejects the next proposed 5."""

    def __init__(self) -> None:
        import torch

        self.device = torch.device("cpu")

    def parameters(self):
        import torch

        yield torch.zeros(1)

    def __call__(self, input_ids, past_key_values=None, use_cache=False):
        import torch

        batch, length = input_ids.shape
        vocab = 16
        logits = torch.zeros(batch, length, vocab)
        logits[:, :, 5] = 10.0
        if length >= 4:
            logits[:, 3, :] = 0.0
            logits[:, 3, 6] = 10.0
        cached = (past_key_values or 0) + length if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=cached)


class _NoCacheFakeModel(_FakeModel):
    def __call__(self, input_ids, past_key_values=None, use_cache=False):
        out = super().__call__(
            input_ids, past_key_values=past_key_values, use_cache=use_cache
        )
        return SimpleNamespace(logits=out.logits, past_key_values=None)


class _CacheStrictIncrementModel:
    """Model that rejects duplicated cache tokens and predicts 4, 5, 6... after a 3-token prompt."""

    def __init__(self) -> None:
        import torch

        self.device = torch.device("cpu")

    def parameters(self):
        import torch

        yield torch.zeros(1)

    def __call__(self, input_ids, past_key_values=None, use_cache=False):
        import torch

        batch, length = input_ids.shape
        vocab = 16
        past_len = int(past_key_values or 0)
        if past_key_values is not None:
            expected = past_len + 1
            actual = int(input_ids[0, 0].item())
            assert actual == expected, f"expected cached token {expected}, got {actual}"

        logits = torch.zeros(batch, length, vocab)
        for pos in range(length):
            next_id = min(past_len + pos + 2, vocab - 1)
            logits[:, pos, next_id] = 10.0
        cached = past_len + length if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=cached)


class _CroppableCache:
    def __init__(self, length: int) -> None:
        self.length = length
        self.crops: list[int] = []

    def __bool__(self) -> bool:
        return True

    def __int__(self) -> int:
        return self.length

    def crop(self, seq_len: int) -> None:
        self.crops.append(seq_len)
        self.length = seq_len


class _CroppableFakeModel(_FakeModel):
    def __init__(self, next_id: int) -> None:
        super().__init__(next_id)
        self.full_replays = 0
        self.last_cache: _CroppableCache | None = None

    def __call__(self, input_ids, past_key_values=None, use_cache=False):
        out = super().__call__(
            input_ids, past_key_values=past_key_values, use_cache=False
        )
        if past_key_values is None:
            self.full_replays += 1
            past_len = 0
        else:
            past_len = int(past_key_values)
        cache = (
            _CroppableCache(past_len + int(input_ids.shape[1])) if use_cache else None
        )
        self.last_cache = cache
        return SimpleNamespace(logits=out.logits, past_key_values=cache)


def test_default_num_speculative_tokens_from_payload():
    assert default_num_speculative_tokens({"num_speculative_tokens": 3}) == 3


def test_default_num_speculative_tokens_env_fallback(monkeypatch):
    monkeypatch.setenv("SEISO_SPECULATIVE_TOKENS", "7")
    assert default_num_speculative_tokens({}) == 7


def test_decode_new_text_is_bpe_safe():
    import torch

    tok = _BpeLikeTokenizer()
    tok._tokens = [10, 11]
    ids = torch.tensor([[10, 11]], dtype=torch.long)

    full_text = tok.decode(ids[0])
    assert full_text == "AB"

    suffix, end = _decode_new_text(tok, ids, prev_char_len=1)
    assert suffix == "B"
    assert end == len(full_text)

    wrong_incremental = tok.decode([11])
    assert wrong_incremental == "X"
    assert wrong_incremental != suffix


def test_iter_speculative_tokens_uses_kv_cache_by_default():
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
            max_new_tokens=2,
            num_speculative_tokens=2,
        )
    )

    assert chunks
    assert any("4" in chunk.text for chunk in chunks)


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
    assert any("4" in chunk.text for chunk in chunks)


def test_cached_and_naive_paths_match(monkeypatch):
    monkeypatch.delenv("SEISO_SPECULATIVE_KV_CACHE", raising=False)
    tok = _FakeTokenizer()
    bundle = TorchSpeculativeBundle(
        target_model=_FakeModel(4),
        target_tokenizer=tok,
        draft_model=_FakeModel(4),
        draft_tokenizer=tok,
    )
    kwargs = {
        "bundle": bundle,
        "prompt": "hello",
        "max_new_tokens": 4,
        "num_speculative_tokens": 2,
    }

    cached = "".join(chunk.text for chunk in iter_speculative_tokens(**kwargs))

    monkeypatch.setenv("SEISO_SPECULATIVE_KV_CACHE", "false")
    naive = "".join(chunk.text for chunk in iter_speculative_tokens(**kwargs))

    assert cached == naive


def test_partial_rejection_still_streams():
    tok = _FakeTokenizer()
    bundle = TorchSpeculativeBundle(
        target_model=_RejectingFakeModel(),
        target_tokenizer=tok,
        draft_model=_FakeModel(5),
        draft_tokenizer=tok,
    )

    chunks = list(
        iter_speculative_tokens(
            bundle=bundle,
            prompt="hello",
            max_new_tokens=2,
            num_speculative_tokens=2,
        )
    )

    assert chunks
    assert any("5" in chunk.text or "6" in chunk.text for chunk in chunks)


def test_cached_rejection_crops_draft_cache_instead_of_replaying_prefix():
    tok = _FakeTokenizer()
    draft = _CroppableFakeModel(5)
    bundle = TorchSpeculativeBundle(
        target_model=_RejectingFakeModel(),
        target_tokenizer=tok,
        draft_model=draft,
        draft_tokenizer=tok,
    )

    chunks = list(
        iter_speculative_tokens(
            bundle=bundle,
            prompt="hello",
            max_new_tokens=2,
            num_speculative_tokens=2,
        )
    )

    assert chunks
    assert draft.full_replays == 1


def test_dflash_cached_reuses_incremental_prompt_text(monkeypatch):
    tok = _FakeTokenizer()
    prompts: list[str] = []

    def _fake_propose(_draft_llm, _target_tok, current_text, k, temperature=0.0):
        prompts.append(current_text)
        return [4] * k

    monkeypatch.setattr(
        "seiso.inference.speculative._propose_with_dflash_draft",
        _fake_propose,
    )
    bundle = DFlashDraftSpeculativeBundle(
        target_model=_FakeModel(4),
        target_tokenizer=tok,
        draft_llm=object(),
        draft_tokenizer=tok,
    )

    chunks = list(
        iter_speculative_tokens_dflash(
            bundle=bundle,
            prompt="hello",
            max_new_tokens=2,
            num_speculative_tokens=1,
        )
    )

    assert chunks
    assert prompts == ["1 2 3", "1 2 3 4"]


def test_cached_draft_does_not_replay_prompt_token():
    tok = _FakeTokenizer()
    bundle = TorchSpeculativeBundle(
        target_model=_CacheStrictIncrementModel(),
        target_tokenizer=tok,
        draft_model=_CacheStrictIncrementModel(),
        draft_tokenizer=tok,
    )

    chunks = list(
        iter_speculative_tokens(
            bundle=bundle,
            prompt="hello",
            max_new_tokens=2,
            num_speculative_tokens=2,
        )
    )

    assert "".join(chunk.text for chunk in chunks) == " 4 5"


def test_kv_cache_falls_back_when_past_key_values_missing(monkeypatch):
    monkeypatch.delenv("SEISO_SPECULATIVE_KV_CACHE", raising=False)
    tok = _FakeTokenizer()
    bundle = TorchSpeculativeBundle(
        target_model=_NoCacheFakeModel(4),
        target_tokenizer=tok,
        draft_model=_FakeModel(4),
        draft_tokenizer=tok,
    )

    chunks = list(
        iter_speculative_tokens(
            bundle=bundle,
            prompt="hello",
            max_new_tokens=2,
            num_speculative_tokens=2,
        )
    )

    assert chunks


@pytest.mark.asyncio
async def test_runner_routes_to_speculative_stream(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    seen: dict[str, str] = {}

    async def _noop_switch(
        _path: str, *, draft_path: str | None = None, route: str | None = None
    ) -> None:
        seen["draft_path"] = draft_path or ""

    from seiso.inference.streaming import StreamToken

    def _fake_speculative(_payload, _model_path, should_stop):
        assert not should_stop()
        yield StreamToken("spec")

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


def test_iter_speculative_tokens_kv_cache_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SEISO_SPECULATIVE_KV_CACHE", "false")
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
            max_new_tokens=2,
            num_speculative_tokens=2,
        )
    )

    assert chunks


def test_dflash_draft_infer_serializes_concurrent_calls(monkeypatch):
    from seiso.inference.model_pool import DflashDraftHandle, dflash_draft_infer

    active = {"count": 0, "max": 0}

    class _FakeCtx:
        def kv_cache_seq_rm(self, _seq: int, start: int, end: int) -> bool:
            return True

    class _FakeDraftLlm:
        def __init__(self) -> None:
            self.n_tokens = 0
            self.input_ids = [0] * 128
            self._ctx = _FakeCtx()
            self._requires_eval = False
            self._model = type("M", (), {"vocab": object()})()

        @property
        def eval_tokens(self):
            from collections import deque

            return deque(self.input_ids[: self.n_tokens])

        def reset(self) -> None:
            self.n_tokens = 0

        def tokenize(self, text: bytes, add_bos: bool = False, special: bool = False):
            return [1] * max(1, len(text.decode().split()))

        def detokenize(self, tokens, prev_tokens=None):
            return b"x" * len(tokens)

        def eval(self, tokens) -> None:
            import time

            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
            time.sleep(0.05)
            self.n_tokens += len(tokens)
            active["count"] -= 1

        def generate(self, tokens, temp=0.0, top_k=1, reset=True):
            tok = 99
            self.input_ids[self.n_tokens] = tok
            self.n_tokens += 1
            yield tok

    monkeypatch.setattr(
        "llama_cpp.llama_cpp.llama_vocab_is_eog", lambda _vocab, _tok: False
    )
    handle = DflashDraftHandle(_FakeDraftLlm())
    errors: list[BaseException] = []

    def _run() -> None:
        try:
            assert dflash_draft_infer(handle, "prompt", max_tokens=1) == "x"
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert active["max"] == 1


def test_dflash_session_reuses_prefix_eval(monkeypatch):
    from seiso.inference.model_pool import DflashDraftHandle, DflashDraftSession

    monkeypatch.setattr(
        "llama_cpp.llama_cpp.llama_vocab_is_eog", lambda _vocab, _tok: False
    )

    class _FakeCtx:
        def kv_cache_seq_rm(self, _seq: int, start: int, end: int) -> bool:
            return True

    class _FakeDraftLlm:
        def __init__(self) -> None:
            self.n_tokens = 0
            self.input_ids = [10, 11, 12, 13, 14]
            self._ctx = _FakeCtx()
            self._requires_eval = False
            self._model = type("M", (), {"vocab": object()})()
            self.eval_calls: list[list[int]] = []

        @property
        def eval_tokens(self):
            from collections import deque

            return deque(self.input_ids[: self.n_tokens])

        def reset(self) -> None:
            self.n_tokens = 0

        def tokenize(self, text: bytes, add_bos: bool = False, special: bool = False):
            mapping = {b"a": [10], b"a b": [10, 11], b"a b c": [10, 11, 12]}
            return list(mapping.get(text, [10]))

        def detokenize(self, tokens, prev_tokens=None):
            return b"c" * len(tokens)

        def eval(self, tokens) -> None:
            if tokens:
                self.eval_calls.append(list(tokens))
            self.n_tokens += len(tokens)

        def generate(self, tokens, temp=0.0, top_k=1, reset=True):
            tok = 12
            self.input_ids[self.n_tokens] = tok
            self.n_tokens += 1
            yield tok

    llm = _FakeDraftLlm()
    session = DflashDraftSession(DflashDraftHandle(llm))

    assert session.propose("a", max_tokens=1) == "c"
    assert session.propose("a b", max_tokens=1) == "c"
    assert llm.eval_calls == [[10], [11]]


def test_dflash_session_propose_token_ids_skips_text_roundtrip(monkeypatch):
    from seiso.inference.model_pool import DflashDraftHandle, DflashDraftSession

    monkeypatch.setattr(
        "llama_cpp.llama_cpp.llama_vocab_is_eog", lambda _vocab, _tok: False
    )

    class _FakeCtx:
        def kv_cache_seq_rm(self, _seq: int, start: int, end: int) -> bool:
            return True

    class _FakeDraftLlm:
        def __init__(self) -> None:
            self.n_tokens = 0
            self.input_ids = [10, 11, 12, 99]
            self._ctx = _FakeCtx()
            self._requires_eval = False
            self._model = type("M", (), {"vocab": object()})()
            self.eval_calls: list[list[int]] = []

        @property
        def eval_tokens(self):
            from collections import deque

            return deque(self.input_ids[: self.n_tokens])

        def reset(self) -> None:
            self.n_tokens = 0

        def tokenize(self, text: bytes, add_bos: bool = False, special: bool = False):
            return [10, 11] if text == b"hi" else [10]

        def eval(self, tokens) -> None:
            if tokens:
                self.eval_calls.append(list(tokens))
            self.n_tokens += len(tokens)

        def generate(self, tokens, temp=0.0, top_k=1, reset=True):
            tok = 99
            if self.n_tokens < len(self.input_ids):
                self.input_ids[self.n_tokens] = tok
            else:
                self.input_ids.append(tok)
            self.n_tokens += 1
            yield tok

    llm = _FakeDraftLlm()
    session = DflashDraftSession(DflashDraftHandle(llm))

    assert session.propose_token_ids("hi", max_tokens=1) == [99]
    eval_before = len(llm.eval_calls)
    assert session.propose_token_ids("hi", max_tokens=1) == [99]
    assert len(llm.eval_calls) == eval_before


def test_unload_all_clears_dflash_draft_cache(monkeypatch, tmp_path):
    from seiso.inference import model_pool as mp

    closed = {"count": 0}

    class _FakeLlama:
        def close(self) -> None:
            closed["count"] += 1

    norm = str((tmp_path / "draft.gguf").resolve())
    mp._dflash_draft_cache[norm] = mp.DflashDraftHandle(_FakeLlama())

    pool = mp.ModelPool()
    pool.unload_all()

    assert closed["count"] == 1
    assert mp._dflash_draft_cache == {}


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
