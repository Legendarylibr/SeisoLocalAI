"""Tests for VRAM model pool."""

import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from seiso.inference.model_pool import BackendKind, ModelPool, llama_load_kwargs


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


def test_switch_serializes_concurrent_loads_for_same_model(tmp_path):
    pool = ModelPool()
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    handle = object()
    load_count = 0
    count_lock = threading.Lock()
    start = threading.Barrier(6)

    def loader(path: str) -> object:
        nonlocal load_count
        assert path == str(model_path.absolute())
        with count_lock:
            load_count += 1
        time.sleep(0.05)
        return handle

    def switch_once() -> object:
        start.wait()
        return pool.switch(str(model_path), BackendKind.LLAMA, loader)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(switch_once) for _ in range(5)]
        start.wait()
        results = [future.result(timeout=2) for future in futures]

    assert results == [handle] * 5
    assert load_count == 1


def test_switching_gguf_models_closes_previous_handle(tmp_path):
    pool = ModelPool()
    first_path = tmp_path / "first.gguf"
    second_path = tmp_path / "second.gguf"
    first_path.write_bytes(b"gguf")
    second_path.write_bytes(b"gguf")

    class FakeLlama:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    first = FakeLlama()
    second = FakeLlama()

    pool.switch(str(first_path), BackendKind.LLAMA, lambda _path: first)
    pool.switch(str(second_path), BackendKind.LLAMA, lambda _path: second)

    assert first.closed is True
    assert second.closed is False
    assert pool.status()["path"] == str(second_path.absolute())


def test_llama_reuses_larger_preloaded_context(monkeypatch, tmp_path):
    from seiso.inference import model_pool

    pool = ModelPool()
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    handle = object()
    load_paths: list[str] = []

    class FakeLlama:
        def __init__(self, model_path: str, **_kwargs):
            load_paths.append(model_path)

    monkeypatch.setattr(model_pool, "llama_load_kwargs", lambda n_ctx: {"n_ctx": n_ctx})
    monkeypatch.setitem(__import__("sys").modules, "llama_cpp", type("LlamaModule", (), {"Llama": FakeLlama}))
    monkeypatch.setattr(
        "seiso.inference.tuning.attach_llama_prompt_cache",
        lambda _llm: None,
    )

    first = pool.get_llama(str(model_path), n_ctx=4096)
    pool._active.handle = handle
    second = pool.get_llama(str(model_path), n_ctx=2048)

    assert first is not None
    assert second is handle
    assert load_paths == [str(model_path.absolute())]


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
