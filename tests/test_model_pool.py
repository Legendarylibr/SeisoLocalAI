"""Tests for VRAM model pool."""

import platform

from seiso.inference.model_pool import ModelPool, llama_load_kwargs


def test_pool_singleton():
    a = ModelPool.get()
    b = ModelPool.get()
    assert a is b


def test_unload_clears_active():
    pool = ModelPool.get()
    pool.unload_all()
    assert pool.active_key is None
    status = pool.status()
    assert status["active_model"] is None


def test_generation_invalidation():
    pool = ModelPool.get()
    gen_a = pool.bump_generation()
    assert pool.is_generation_active(gen_a)
    gen_b = pool.bump_generation()
    assert not pool.is_generation_active(gen_a)
    assert pool.is_generation_active(gen_b)


def test_cancel_and_unload_clears_active():
    pool = ModelPool.get()
    pool.cancel_and_unload()
    assert pool.active_key is None


def test_llama_load_kwargs_are_tuned_and_overrideable(monkeypatch):
    monkeypatch.setenv("SEISO_LLAMA_THREADS", "6")
    monkeypatch.setenv("SEISO_LLAMA_GPU_LAYERS", "4")
    monkeypatch.setenv("SEISO_LLAMA_USE_MMAP", "false")

    kwargs = llama_load_kwargs(2048)

    assert kwargs["n_ctx"] == 2048
    assert kwargs["n_threads"] == 6
    assert kwargs["n_threads_batch"] == 6
    assert kwargs["n_gpu_layers"] == 4
    assert kwargs["use_mmap"] is False
    assert kwargs["verbose"] is False


def test_llama_load_kwargs_default_metal_offload_on_apple_silicon(monkeypatch):
    monkeypatch.delenv("SEISO_LLAMA_GPU_LAYERS", raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    kwargs = llama_load_kwargs(4096)
    assert kwargs["n_gpu_layers"] == -1
    assert kwargs["flash_attn"] is True
    assert kwargs["no_perf"] is True


def test_llama_load_kwargs_cuda_defaults(monkeypatch):
    monkeypatch.delenv("SEISO_LLAMA_GPU_LAYERS", raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr("seiso.inference.model_pool._cuda_available", lambda: True)

    kwargs = llama_load_kwargs(4096)
    assert kwargs["n_gpu_layers"] == -1
    assert kwargs["n_batch"] == 2048
    assert kwargs["flash_attn"] is True
    assert kwargs["offload_kqv"] is True
    assert kwargs["op_offload"] is True
