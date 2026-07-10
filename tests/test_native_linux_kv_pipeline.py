"""Deterministic coverage for the compatibility-first native Linux KV pipeline."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class _Tokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(int(item)) for item in ids)


class _LengthModel:
    """Returns a cache length and predicts length + 1."""

    def __init__(self, *, max_chunk: int | None = None):
        self.max_chunk = max_chunk
        self.call_sizes: list[int] = []

    def __call__(self, input_ids, *, past_key_values=None, **_kwargs):
        import torch

        size = int(input_ids.shape[-1])
        self.call_sizes.append(size)
        if self.max_chunk is not None and size > self.max_chunk:
            raise torch.OutOfMemoryError("synthetic prefill pressure")
        previous = int(past_key_values or 0)
        total = previous + size
        logits = torch.zeros(1, size, 100)
        logits[..., min(98, total + 1)] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=total)


def _run(model, input_ids, **kwargs):
    import torch

    from seiso.inference.torch_stream import iter_torch_kv_tokens

    return list(
        iter_torch_kv_tokens(
            model=model,
            tokenizer=_Tokenizer(),
            input_ids=torch.tensor([input_ids]),
            max_new_tokens=1,
            **kwargs,
        )
    )


def test_chunked_prefill_matches_one_shot():
    from seiso.inference.torch_stream import clear_torch_prefix_cache

    clear_torch_prefix_cache()
    one_shot = _run(_LengthModel(), list(range(8)), prefill_chunk_size=8)
    chunked_model = _LengthModel()
    chunked = _run(chunked_model, list(range(8)), prefill_chunk_size=2)

    assert [part.text for part in chunked] == [part.text for part in one_shot]
    assert chunked_model.call_sizes[:4] == [2, 2, 2, 2]


def test_exact_prefix_reuse_and_nonprefix_invalidation():
    from seiso.inference.torch_stream import clear_torch_prefix_cache

    clear_torch_prefix_cache()
    model = _LengthModel()
    first_stats: dict = {}
    _run(
        model,
        [1, 2],
        prefill_chunk_size=2,
        cache_key="model",
        prefix_cache=True,
        stats=first_stats,
    )
    assert first_stats["prefix_hit"] is False

    second_stats: dict = {}
    _run(
        model,
        [1, 2, 3, 7],
        prefill_chunk_size=2,
        cache_key="model",
        prefix_cache=True,
        stats=second_stats,
    )
    assert second_stats["prefix_hit"] is True

    miss_stats: dict = {}
    _run(
        model,
        [8, 9],
        prefill_chunk_size=2,
        cache_key="model",
        prefix_cache=True,
        stats=miss_stats,
    )
    assert miss_stats["prefix_hit"] is False


def test_prefill_pressure_halves_chunk_without_replay():
    stats: dict = {}
    model = _LengthModel(max_chunk=2)
    chunks = _run(
        model,
        list(range(8)),
        prefill_chunk_size=8,
        stats=stats,
    )

    assert len(chunks) == 1
    assert stats["prefill_backoffs"] == 2
    assert stats["prefill_chunk_size"] == 2


def test_cancellation_does_not_publish_prefix():
    from seiso.inference.torch_stream import clear_torch_prefix_cache

    clear_torch_prefix_cache()
    _run(
        _LengthModel(),
        [1, 2],
        cache_key="cancel-model",
        prefix_cache=True,
        should_stop=lambda: True,
    )
    stats: dict = {}
    _run(
        _LengthModel(),
        [1, 2],
        cache_key="cancel-model",
        prefix_cache=True,
        stats=stats,
    )
    assert stats["prefix_hit"] is False


def test_pool_unload_invalidates_prefix_state():
    from seiso.inference.model_pool import ModelPool
    from seiso.inference.torch_stream import clear_torch_prefix_cache

    clear_torch_prefix_cache()
    model = _LengthModel()
    _run(model, [1, 2], cache_key="pool-model", prefix_cache=True)
    ModelPool().cancel_and_unload()

    stats: dict = {}
    _run(
        model,
        [1, 2, 3],
        cache_key="pool-model",
        prefix_cache=True,
        stats=stats,
    )
    assert stats["prefix_hit"] is False


def test_policy_rejects_static_cache_without_headroom(monkeypatch):
    from seiso.inference.kv_policy import resolve_kv_cache_policy

    monkeypatch.setenv("SEISO_TORCH_KV_HEADROOM_MB", "0")
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=32,
            num_key_value_heads=8,
            num_attention_heads=32,
            hidden_size=4096,
        )
    )
    policy = resolve_kv_cache_policy(
        {"cache_implementation": "static"},
        model=model,
        input_tokens=4096,
        max_tokens=512,
        free_mb=128,
    )
    assert policy.cache_implementation == "dynamic"
    assert policy.fallback_reason


def test_quantized_cache_requires_opt_in(monkeypatch):
    from seiso.inference.kv_policy import resolve_kv_cache_policy

    monkeypatch.delenv("SEISO_TORCH_QUANTIZED_KV", raising=False)
    implicit = resolve_kv_cache_policy(
        {"kv_policy": {"cache_implementation": "quantized"}},
        model=object(),
        input_tokens=10,
        max_tokens=10,
        free_mb=4096,
    )
    explicit = resolve_kv_cache_policy(
        {"cache_implementation": "quantized"},
        model=object(),
        input_tokens=10,
        max_tokens=10,
        free_mb=4096,
    )
    assert implicit.cache_implementation == "dynamic"
    assert explicit.cache_implementation == "quantized"


def test_generate_retries_unsupported_cache_mode():
    from seiso.inference.tuning import generate_with_cache_fallback

    class _Model:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            if "cache_implementation" in kwargs:
                raise ValueError("cache_implementation is not supported")
            return "ok"

    model = _Model()
    assert (
        generate_with_cache_fallback(
            model,
            {"cache_implementation": "static", "cache_config": {"nbits": 8}},
        )
        == "ok"
    )
    assert "cache_implementation" not in model.calls[-1]
    assert "cache_config" not in model.calls[-1]


def test_cache_mode_fallback_is_blocked_after_stream_emission():
    from seiso.inference.tuning import generate_with_cache_fallback

    class _Model:
        calls = 0

        def generate(self, **_kwargs):
            self.calls += 1
            raise ValueError("cache_implementation is not supported")

    model = _Model()
    with pytest.raises(ValueError):
        generate_with_cache_fallback(
            model,
            {"cache_implementation": "static"},
            can_retry=lambda: False,
        )
    assert model.calls == 1


def test_decode_compile_is_guarded_by_successful_warmup(monkeypatch):
    import torch

    from seiso.inference.tuning import maybe_compile_torch_decode

    monkeypatch.setenv("SEISO_TORCH_DECODE_GRAPHS", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    compile_calls: list[dict] = []

    def fake_compile(function, **kwargs):
        compile_calls.append(kwargs)
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)

    class _Model:
        def __init__(self):
            self.forward = self._forward

        def _forward(self, input_ids, **_kwargs):
            return SimpleNamespace(
                logits=torch.zeros(1, input_ids.shape[-1], 4),
                past_key_values=object(),
            )

        def __call__(self, **kwargs):
            return self.forward(**kwargs)

    model = _Model()
    assert maybe_compile_torch_decode(model, torch.tensor([[1, 2]])) is True
    assert compile_calls
    assert model._seiso_decode_compiled is True


def test_sidecar_options_are_capability_gated(monkeypatch):
    from seiso.inference import llamaswap

    monkeypatch.setattr(llamaswap, "plan_sidecar_request", lambda *_a: ([], 4096, 32))
    monkeypatch.setattr(llamaswap, "sidecar_ollama_num_batch", lambda: None)
    monkeypatch.setattr(llamaswap, "sidecar_ollama_num_gpu", lambda *_a, **_k: None)
    monkeypatch.setattr(llamaswap, "sidecar_ollama_keep_alive", lambda **_k: None)
    ollama = llamaswap.OllamaClient()
    monkeypatch.setattr(ollama, "_resolve_model", lambda *_a: "model")

    baseline = ollama._request_body({}, "/tmp/m.gguf", stream=True)
    negotiated = ollama._request_body(
        {"sidecar_capabilities": ["num_keep"], "sidecar_num_keep": 64},
        "/tmp/m.gguf",
        stream=True,
    )
    assert "num_keep" not in baseline["options"]
    assert negotiated["options"]["num_keep"] == 64

    swap = llamaswap.LlamaSwapClient()
    baseline_swap = swap._request_body({}, "/tmp/m.gguf", stream=True)
    negotiated_swap = swap._request_body(
        {"sidecar_capabilities": ["cache_prompt"]}, "/tmp/m.gguf", stream=True
    )
    assert "cache_prompt" not in baseline_swap
    assert negotiated_swap["cache_prompt"] is True
