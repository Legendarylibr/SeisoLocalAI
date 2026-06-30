"""Tests for cross-cutting memory protection."""

from __future__ import annotations

import pytest

from seiso.memory.protection import (
    MemoryLoadBlockedError,
    apply_rl_memory_guards,
    apply_training_memory_guards,
    assess_path_memory_fit,
    clamp_llama_cache_mb,
    clamp_llama_load_kwargs,
    clamp_llama_n_ctx,
    ensure_load_fits,
    is_oom_error,
    sanitize_inference_payload,
)


def test_is_oom_error_detects_cuda_message():
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))


def test_is_oom_error_ignores_other_errors():
    assert not is_oom_error(ValueError("bad batch"))


def test_sanitize_inference_payload_clamps_max_tokens(monkeypatch):
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 4096)
    out = sanitize_inference_payload(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 99999}
    )
    assert 1 <= out["max_tokens"] <= 8192


def test_clamp_llama_n_ctx_respects_headroom(monkeypatch):
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 3072)
    n_ctx = clamp_llama_n_ctx(
        8192, messages=[{"role": "user", "content": "x" * 200}], max_tokens=256
    )
    assert 2048 <= n_ctx <= 8192
    assert n_ctx % 512 == 0


def test_clamp_llama_load_kwargs_reduces_batch_on_tight_memory(monkeypatch):
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 3500)
    kwargs = clamp_llama_load_kwargs(
        {"n_ctx": 4096, "n_batch": 2048, "n_ubatch": 2048, "n_gpu_layers": -1}
    )
    assert kwargs["n_batch"] <= 512
    assert kwargs["n_ubatch"] <= kwargs["n_batch"]


def test_clamp_llama_load_kwargs_scales_batch_with_large_context(monkeypatch):
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 16384)
    kwargs = clamp_llama_load_kwargs(
        {"n_ctx": 8192, "n_batch": 2048, "n_ubatch": 512, "n_gpu_layers": -1}
    )
    assert kwargs["n_batch"] <= 1024


def test_llama_batch_headroom_accounts_for_model_weights(monkeypatch, tmp_path):
    from seiso.memory.protection import llama_batch_headroom_mb

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\x00" * (8 * 1024**2))
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb",
        lambda _path: 8192,
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_block_count",
        lambda _path: 32,
    )
    remaining = llama_batch_headroom_mb(16384, model_path=gguf, n_gpu_layers=-1)
    assert remaining < 8192


def test_clamp_llama_cache_mb_disabled_on_low_headroom(monkeypatch):
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 2048)
    assert clamp_llama_cache_mb(1024) == 0


def test_apply_training_memory_guards_clamps_batch(monkeypatch):
    from seiso.training.config import TrainConfig

    profile = {
        "backend": "cuda",
        "gpus": [{"vram_total_mb": 6000, "vram_used_mb": 1000}],
        "ram_gb": 16,
    }
    monkeypatch.setattr("seiso.memory.protection.hardware_profile", lambda: profile)
    monkeypatch.setattr("forge.services.hardware.vram_headroom_mb", lambda _p: 5000)

    cfg = TrainConfig(
        model_id="meta-llama/Llama-3.2-1B",
        dataset="data.jsonl",
        batch_size=8,
        gradient_accumulation_steps=1,
        max_seq_length=8192,
    )
    guarded = apply_training_memory_guards(cfg)
    assert guarded.batch_size <= 2
    assert guarded.max_seq_length <= 4096


def test_apply_rl_memory_guards_scales_preflight(monkeypatch):
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 2048)
    out = apply_rl_memory_guards(
        {
            "torch_preflight_batch_size": 16384,
            "replay_buffer_on_gpu": True,
            "torch_batch_episodes": 2048,
        }
    )
    assert out["torch_preflight_batch_size"] <= 768
    assert out["replay_buffer_on_gpu"] is False
    assert out["torch_batch_episodes"] <= 384


def test_ensure_load_fits_blocks_oversized_gguf(tmp_path, monkeypatch):
    gguf = tmp_path / "huge.gguf"
    gguf.write_bytes(b"\x00" * (9 * 1024**3))
    profile = {
        "backend": "cuda",
        "gpus": [{"vram_total_mb": 4096, "vram_used_mb": 0}],
        "ram_gb": 16,
    }
    monkeypatch.setattr(
        "seiso.memory.protection.hardware_profile", lambda force_refresh=False: profile
    )
    monkeypatch.setattr("seiso.hardware.fit.fit_headroom_mb", lambda _p: 2048)
    monkeypatch.setattr("seiso.hardware.fit.vram_headroom_mb", lambda _p: 2048)
    monkeypatch.setattr(
        "seiso.inference.model_pool.ModelPool.prepare_for_load",
        lambda self, *args, **kwargs: False,
    )
    with pytest.raises(MemoryLoadBlockedError):
        ensure_load_fits(gguf, mode="chat")


def test_ensure_load_fits_forwards_backend_to_pool(tmp_path, monkeypatch):
    from seiso.inference.model_pool import ModelPool

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\x00")
    calls: list[tuple[str, str | None]] = []
    pool = ModelPool()

    monkeypatch.setattr(
        "seiso.memory.protection.assess_path_memory_fit",
        lambda _path, mode="chat": {"memory_load_blocked": False},
    )
    monkeypatch.setattr(
        "seiso.inference.model_pool.ModelPool.prepare_for_load",
        lambda self, target_path, backend=None: calls.append((target_path, backend))
        or False,
    )

    monkeypatch.setattr(
        "seiso.memory.protection.assess_path_memory_fit_for_load",
        lambda path, mode="chat", backend=None: (
            pool.prepare_for_load(str(path), backend) or {"memory_load_blocked": False}
        ),
    )

    ensure_load_fits(gguf, mode="chat", backend="llamacpp")

    assert calls == [(str(gguf), "llamacpp")]


def test_assess_path_memory_fit_for_small_file(tmp_path, monkeypatch):
    gguf = tmp_path / "tiny.gguf"
    gguf.write_bytes(b"\x00" * (32 * 1024**2))
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 16384)
    fit = assess_path_memory_fit(gguf, mode="chat")
    assert fit.get("memory_load_blocked") is False


def test_allow_memory_overcommit_skips_block(tmp_path, monkeypatch):
    gguf = tmp_path / "big.gguf"
    gguf.write_bytes(b"\x00" * (4 * 1024**3))
    monkeypatch.setattr(
        "seiso.memory.protection.assess_path_memory_fit",
        lambda _path, mode="chat": {
            "memory_load_blocked": True,
            "memory_load_blocked_reason": "Model exceeds available memory",
            "hardware_fit": "unlikely",
        },
    )
    monkeypatch.setenv("SEISO_ALLOW_MEMORY_OVERCOMMIT", "1")
    fit = ensure_load_fits(gguf, mode="chat")
    assert fit.get("memory_load_blocked") is True
