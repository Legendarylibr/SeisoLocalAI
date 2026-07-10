"""Tests for inference benchmark helpers."""

from __future__ import annotations

import pytest

from seiso.inference.benchmark import (
    BASELINE_ENV,
    InferenceBenchResult,
    _estimate_tokens,
    compare_inference_profiles,
)


@pytest.mark.asyncio
async def test_bench_inference_with_mocked_runner(monkeypatch):
    from seiso.inference import benchmark as bench_mod

    class FakeRunner:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, _payload):
            self.calls += 1
            return "warm"

        async def stream_updates(self, _payload):
            from seiso.inference.streaming import StreamUpdate

            yield StreamUpdate(text="Hello", output_tokens=1)
            yield StreamUpdate(text=" world", output_tokens=2)

    fake = FakeRunner()
    monkeypatch.setattr(bench_mod, "LocalInferenceRunner", lambda: fake)

    class FakePool:
        def cancel_and_unload(self) -> None:
            return None

    monkeypatch.setattr(bench_mod, "get_model_pool", lambda: FakePool())
    monkeypatch.setattr(
        bench_mod,
        "resolve_local_backend",
        lambda **_kwargs: "llamacpp",
    )

    result = await bench_mod.bench_inference(
        "/tmp/model.gguf",
        prompt="hi",
        max_tokens=32,
        backend="llamacpp",
        warmup=True,
    )

    assert isinstance(result, InferenceBenchResult)
    assert result.output_chars == len("Hello world")
    assert result.output_tokens == 2
    assert result.backend == "llamacpp"
    assert fake.calls == 1


@pytest.mark.asyncio
async def test_kv_benchmark_covers_short_long_and_repeated(monkeypatch):
    from seiso.inference import benchmark as bench_mod

    class FakeRunner:
        def __init__(self) -> None:
            self.payloads = []
            self.last_inference_stats = {"prefill_chunks": 2, "prefix_hit": False}

        async def stream_updates(self, payload):
            from seiso.inference.streaming import StreamUpdate

            self.payloads.append(payload)
            self.last_inference_stats = {
                "prefill_chunks": 2,
                "prefix_hit": len(payload.get("messages", [])) > 1,
            }
            yield StreamUpdate(text="ok", output_tokens=1)

    fake = FakeRunner()
    monkeypatch.setattr(bench_mod, "LocalInferenceRunner", lambda: fake)
    monkeypatch.setattr(
        bench_mod, "resolve_local_backend", lambda **_kwargs: "torch"
    )
    monkeypatch.setattr(
        bench_mod,
        "get_model_pool",
        lambda: type("Pool", (), {"cancel_and_unload": lambda self: None})(),
    )

    report = await bench_mod.benchmark_kv_scenarios(
        "/tmp/model", short_prompt="short", long_prompt="long"
    )

    assert set(report) == {
        "short",
        "long",
        "repeated_first",
        "repeated_second",
    }
    assert report["repeated_second"]["kv_metadata"]["prefix_hit"] is True
    assert len(fake.payloads[-1]["messages"]) == 3


@pytest.mark.asyncio
async def test_compare_profiles_runs_twice(monkeypatch):
    from seiso.inference import benchmark as bench_mod

    calls: list[str] = []

    async def fake_bench(*_args, profile="optimized", env_overrides=None, **_kwargs):
        calls.append(profile)
        return InferenceBenchResult(
            backend="llamacpp",
            model_path="/tmp/model.gguf",
            prompt_chars=2,
            output_chars=10,
            output_tokens=5,
            max_tokens=32,
            load_ms=100.0,
            ttft_ms=50.0,
            generate_ms=200.0,
            total_ms=350.0,
            tokens_per_sec=25.0 if profile == "optimized" else 10.0,
            ms_per_token=40.0 if profile == "optimized" else 100.0,
            profile=profile,
        )

    monkeypatch.setattr(bench_mod, "bench_inference", fake_bench)

    class FakePool:
        def cancel_and_unload(self) -> None:
            return None

    monkeypatch.setattr(bench_mod, "get_model_pool", lambda: FakePool())

    report = await compare_inference_profiles(
        "/tmp/model.gguf", prompt="hi", max_tokens=32
    )
    assert calls == ["baseline", "optimized"]
    assert report["speedup_tokens_per_sec"] == 2.5
    assert report["baseline"]["profile"] == "baseline"
    assert report["optimized"]["profile"] == "optimized"


def test_estimate_tokens():
    assert _estimate_tokens("") == 0
    assert _estimate_tokens("one two three four") >= 4


def test_baseline_env_disables_gpu_offload():
    assert BASELINE_ENV["SEISO_LLAMA_GPU_LAYERS"] == "0"
    assert BASELINE_ENV["SEISO_INFERENCE_FUSED_KERNELS"] == "false"


def test_cli_benchmark_uses_sync_wrappers(monkeypatch):
    from seiso.inference import benchmark as bench_mod
    from seiso_cli import main as cli_mod

    calls: list[tuple[str, dict]] = []
    result = InferenceBenchResult(
        backend="llamacpp",
        model_path="/tmp/model.gguf",
        prompt_chars=2,
        output_chars=5,
        output_tokens=2,
        max_tokens=8,
        load_ms=10.0,
        ttft_ms=5.0,
        generate_ms=20.0,
        total_ms=35.0,
        tokens_per_sec=100.0,
        ms_per_token=10.0,
    )

    def fake_run_bench_inference(**kwargs):
        calls.append(("bench", kwargs))
        return result

    def fake_run_compare_inference_profiles(**kwargs):
        calls.append(("compare", kwargs))
        return {
            "baseline": {**result.to_dict(), "profile": "baseline"},
            "optimized": {**result.to_dict(), "profile": "optimized"},
            "speedup_tokens_per_sec": 2.0,
            "ttft_improvement_ms": 5.0,
        }

    monkeypatch.setattr(bench_mod, "run_bench_inference", fake_run_bench_inference)
    monkeypatch.setattr(
        bench_mod,
        "run_compare_inference_profiles",
        fake_run_compare_inference_profiles,
    )

    cli_mod.bench_inference_cmd(
        model="/tmp/model.gguf",
        prompt="hi",
        max_tokens=8,
        backend="llamacpp",
        compare=False,
        json_out=True,
    )
    cli_mod.bench_inference_cmd(
        model="/tmp/model.gguf",
        prompt="hi",
        max_tokens=8,
        backend="llamacpp",
        compare=True,
        json_out=True,
    )

    assert calls == [
        (
            "bench",
            {
                "model_path": "/tmp/model.gguf",
                "prompt": "hi",
                "max_tokens": 8,
                "backend": "llamacpp",
                "warmup": True,
            },
        ),
        (
            "compare",
            {
                "model_path": "/tmp/model.gguf",
                "prompt": "hi",
                "max_tokens": 8,
                "backend": "llamacpp",
            },
        ),
    ]
