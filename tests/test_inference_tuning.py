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
    assert _stream_batch_chars() == 4


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


def test_prepare_torch_model_patches_before_compile_and_returns_wrapper(monkeypatch):
    from types import SimpleNamespace

    from seiso.inference import tuning

    events: list[str] = []

    class _Model:
        config = SimpleNamespace(use_cache=False)

        def eval(self):
            events.append("eval")
            return self

    model = _Model()
    compiled = object()
    monkeypatch.setattr(tuning, "configure_torch_inference", lambda: events.append("configure"))
    monkeypatch.setattr(
        tuning, "apply_inference_kernels", lambda value: events.append(f"kernels:{value is model}")
    )
    monkeypatch.setattr(
        tuning,
        "maybe_compile_torch_model",
        lambda value: events.append(f"compile:{value is model}") or compiled,
    )

    assert tuning.prepare_torch_model(model) is compiled
    assert model.config.use_cache is True
    assert events == ["configure", "eval", "kernels:True", "compile:True"]


def test_prepare_torch_model_restores_kernels_when_compile_raises(monkeypatch):
    from types import SimpleNamespace

    import pytest

    from seiso.inference import tuning

    class _Model:
        config = SimpleNamespace(use_cache=False)

        def eval(self):
            return self

    model = _Model()
    restored: list[object] = []
    monkeypatch.setattr(tuning, "configure_torch_inference", lambda: None)
    monkeypatch.setattr(tuning, "apply_inference_kernels", lambda value: None)
    monkeypatch.setattr(
        tuning,
        "maybe_compile_torch_model",
        lambda value: (_ for _ in ()).throw(RuntimeError("compile failed")),
    )
    monkeypatch.setattr(
        "seiso.kernels.lifecycle.restore_kernel_patches",
        lambda value=None: restored.append(value) or 0,
    )

    with pytest.raises(RuntimeError, match="compile failed"):
        tuning.prepare_torch_model(model)
    assert restored == [model]


def test_apply_inference_kernels_commits_session_on_success(monkeypatch):
    from types import SimpleNamespace

    from seiso.inference import tuning
    from seiso.kernels.lifecycle import KernelPatchSession, _ACTIVE_PATCH_SESSION

    model = SimpleNamespace()
    events: list[str] = []
    commits: list[KernelPatchSession] = []
    original_commit = KernelPatchSession.commit

    def _commit(self):
        commits.append(self)
        return original_commit(self)

    def _fake_apply(m, **_kwargs):
        events.append("apply")
        assert _ACTIVE_PATCH_SESSION.get() is not None
        assert m is model

    class _Cuda:
        @staticmethod
        def is_available():
            return True

    fake_torch = SimpleNamespace(cuda=_Cuda())
    monkeypatch.setattr(
        tuning,
        "env_bool",
        lambda name, default=False: (
            True if name == "SEISO_INFERENCE_FUSED_KERNELS" else default
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(
        "seiso.kernels.attention.enable_torch_sdpa_backends",
        lambda **_k: None,
    )
    monkeypatch.setattr("seiso.kernels.hooks.apply_training_kernels", _fake_apply)
    monkeypatch.setattr(KernelPatchSession, "commit", _commit)

    tuning.apply_inference_kernels(model)
    assert events == ["apply"]
    assert len(commits) == 1
    assert _ACTIVE_PATCH_SESSION.get() is None


def test_apply_inference_kernels_restores_session_on_apply_failure(monkeypatch):
    from types import SimpleNamespace

    from seiso.inference import tuning
    from seiso.kernels.lifecycle import KernelPatchSession, _ACTIVE_PATCH_SESSION

    model = SimpleNamespace()
    restores: list[KernelPatchSession] = []
    original_restore = KernelPatchSession.restore

    def _restore(self):
        restores.append(self)
        return original_restore(self)

    class _Cuda:
        @staticmethod
        def is_available():
            return True

    fake_torch = SimpleNamespace(cuda=_Cuda())
    monkeypatch.setattr(
        tuning,
        "env_bool",
        lambda name, default=False: (
            True if name == "SEISO_INFERENCE_FUSED_KERNELS" else default
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(
        "seiso.kernels.attention.enable_torch_sdpa_backends",
        lambda **_k: None,
    )
    monkeypatch.setattr(
        "seiso.kernels.hooks.apply_training_kernels",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("patch failed")),
    )
    monkeypatch.setattr(KernelPatchSession, "restore", _restore)

    tuning.apply_inference_kernels(model)
    assert len(restores) == 1
    assert _ACTIVE_PATCH_SESSION.get() is None


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
