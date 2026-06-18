"""Tests for inference tuning helpers."""

from __future__ import annotations

from seiso.inference.tuning import (
    extract_mlx_token_text,
    estimate_llama_n_ctx,
    llama_completion_kwargs,
    mlx_stream_kwargs,
    torch_generate_kwargs,
)


class _FakeMlxToken:
    def __init__(self, text: str) -> None:
        self.text = text


def test_extract_mlx_token_text_from_response_object():
    assert extract_mlx_token_text(_FakeMlxToken("hello")) == "hello"
    assert extract_mlx_token_text(_FakeMlxToken("")) is None


def test_mlx_stream_kwargs_greedy_by_default():
    assert mlx_stream_kwargs({"max_tokens": 128}) == {"max_tokens": 128, "prefill_step_size": 4096}


def test_mlx_stream_kwargs_with_temperature():
    pytest = __import__("pytest")
    mlx_lm = pytest.importorskip("mlx_lm")
    _ = mlx_lm  # used for skip only
    kwargs = mlx_stream_kwargs({"max_tokens": 64, "temperature": 0.7, "top_p": 0.9})
    assert kwargs["max_tokens"] == 64
    assert kwargs["sampler"] is not None


def test_torch_generate_kwargs_greedy():
    inputs = {"input_ids": object()}
    streamer = object()
    kwargs = torch_generate_kwargs({"max_tokens": 256, "temperature": 0}, inputs, streamer)
    assert kwargs["do_sample"] is False
    assert kwargs["use_cache"] is True
    assert kwargs["max_new_tokens"] == 256


def test_estimate_llama_n_ctx_sizes_to_prompt():
    messages = [{"role": "user", "content": "x" * 4000}]
    n_ctx = estimate_llama_n_ctx(messages, max_tokens=256)
    assert 2048 <= n_ctx <= 8192
    assert n_ctx % 512 == 0


def test_llama_completion_kwargs_greedy():
    kwargs = llama_completion_kwargs({"max_tokens": 100, "temperature": 0})
    assert kwargs["temperature"] == 0.0
    assert kwargs["stream"] is True
    assert kwargs["max_tokens"] == 100
