"""Tests for inference tuning helpers."""

from __future__ import annotations

from seiso.inference.tuning import (
    estimate_llama_n_ctx,
    extract_mlx_token_text,
    generate_with_cache_fallback,
    llama_completion_kwargs,
    mlx_stream_kwargs,
    torch_generate_kwargs,
)


class _FakeMlxToken:
    def __init__(self, text: str) -> None:
        self.text = text


def test_stream_batch_chars_speed_default(monkeypatch):
    from seiso.inference.runner import _stream_batch_chars

    monkeypatch.delenv("SEISO_STREAM_BATCH_CHARS", raising=False)
    assert _stream_batch_chars() == 16


def test_extract_mlx_token_text_from_response_object():
    assert extract_mlx_token_text(_FakeMlxToken("hello")) == "hello"
    assert extract_mlx_token_text(_FakeMlxToken("")) is None


def test_mlx_stream_kwargs_greedy_by_default():
    assert mlx_stream_kwargs({"max_tokens": 128}) == {
        "max_tokens": 128,
        "prefill_step_size": 4096,
    }


def test_mlx_stream_kwargs_does_not_scale_prefill_by_headroom(monkeypatch):
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 3072)
    assert mlx_stream_kwargs({"max_tokens": 64})["prefill_step_size"] == 4096


def test_mlx_stream_kwargs_with_temperature(monkeypatch):
    monkeypatch.setattr(
        "seiso.inference.tuning.build_mlx_sampler",
        lambda payload: object() if float(payload.get("temperature", 0)) > 0 else None,
    )
    kwargs = mlx_stream_kwargs({"max_tokens": 64, "temperature": 0.7, "top_p": 0.9})
    assert kwargs["max_tokens"] == 64
    assert kwargs["sampler"] is not None


def test_torch_generate_kwargs_greedy():
    inputs = {"input_ids": object()}
    streamer = object()
    kwargs = torch_generate_kwargs(
        {"max_tokens": 256, "temperature": 0}, inputs, streamer
    )
    assert kwargs["do_sample"] is False
    assert kwargs["num_beams"] == 1
    assert kwargs["use_cache"] is True
    assert kwargs["cache_implementation"] == "static"
    assert kwargs["return_dict_in_generate"] is False
    assert kwargs["output_scores"] is False
    assert kwargs["max_new_tokens"] == 256


def test_torch_generate_kwargs_cache_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SEISO_TORCH_CACHE_IMPLEMENTATION", "off")
    kwargs = torch_generate_kwargs({"max_tokens": 32, "temperature": 0}, {}, object())
    assert "cache_implementation" not in kwargs


def test_torch_generate_kwargs_payload_overrides_cache_impl(monkeypatch):
    monkeypatch.setenv("SEISO_TORCH_CACHE_IMPLEMENTATION", "dynamic")
    kwargs = torch_generate_kwargs(
        {"max_tokens": 32, "temperature": 0, "cache_implementation": "static"},
        {},
        object(),
    )
    assert kwargs["cache_implementation"] == "static"


def test_generate_with_cache_fallback_retries_unsupported_cache_impl():
    calls: list[dict] = []

    class _Model:
        def generate(self, **kwargs):
            calls.append(kwargs)
            if "cache_implementation" in kwargs:
                raise ValueError(
                    "The following model_kwargs are not used: ['cache_implementation']"
                )
            return "ok"

    assert (
        generate_with_cache_fallback(_Model(), {"cache_implementation": "dynamic"})
        == "ok"
    )
    assert calls == [{"cache_implementation": "dynamic"}, {}]


def test_generate_with_cache_fallback_keeps_unrelated_errors():
    class _Model:
        def generate(self, **kwargs):
            raise ValueError("bad prompt")

    pytest = __import__("pytest")
    with pytest.raises(ValueError, match="bad prompt"):
        generate_with_cache_fallback(_Model(), {"cache_implementation": "dynamic"})


def test_estimate_llama_n_ctx_sizes_to_prompt(monkeypatch):
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 16384)
    monkeypatch.setattr(
        "seiso.platform.use_linux_nvidia_inference_guards", lambda **_: False
    )
    messages = [{"role": "user", "content": "x" * 4000}]
    n_ctx = estimate_llama_n_ctx(messages, max_tokens=256)
    assert 2048 <= n_ctx <= 131072
    assert n_ctx % 512 == 0


def test_estimate_llama_n_ctx_uses_coarse_buckets(monkeypatch):
    monkeypatch.setattr(
        "seiso.platform.use_linux_nvidia_inference_guards", lambda: False
    )
    short = estimate_llama_n_ctx(
        [{"role": "user", "content": "hi"}], max_tokens=128
    )
    medium = estimate_llama_n_ctx(
        [{"role": "user", "content": "x" * 8000}], max_tokens=256
    )
    assert short == 2048
    # Growing history should jump buckets, not 512-token steps.
    assert medium in (2048, 4096, 8192)
    assert medium >= short


def test_llama_completion_kwargs_greedy():
    kwargs = llama_completion_kwargs({"max_tokens": 100, "temperature": 0})
    assert kwargs["temperature"] == 0.0
    assert kwargs["stream"] is True
    assert kwargs["max_tokens"] == 100
