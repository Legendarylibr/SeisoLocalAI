"""Tests for VRAM model pool."""

import os
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from seiso.inference.model_pool import BackendKind, LoadedModel, ModelPool, llama_load_kwargs


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


def test_prepare_for_load_unloads_when_switching(tmp_path, monkeypatch):
    pool = ModelPool()
    first = tmp_path / "a.gguf"
    second = tmp_path / "b.gguf"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    pool._active = LoadedModel(
        key="llama:a",
        backend=BackendKind.LLAMA,
        handle=object(),
        meta={"path": str(first), "norm_path": str(first.resolve())},
    )
    refreshed = {"calls": 0}

    def _refresh(force_refresh=False):
        refreshed["calls"] += 1
        return {}

    monkeypatch.setattr("seiso.hardware.profile.hardware_profile", _refresh)
    unloaded = pool.prepare_for_load(str(second), BackendKind.LLAMA)
    assert unloaded is True
    assert pool.active_key is None
    assert refreshed["calls"] == 1


def test_llama_gpu_layers_optimal_uses_short_ttl_cache(monkeypatch, tmp_path):
    import seiso.inference.model_pool as mp

    mp._clear_optimal_layers_cache()
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"gguf")
    calls: list[int] = []

    monkeypatch.setattr(mp, "fit_llama_gpu_layers", lambda _p, _r, _h: calls.append(1) or 32)

    first = mp._llama_gpu_layers_optimal(str(gguf), -1)
    second = mp._llama_gpu_layers_optimal(str(gguf), -1)

    assert first == 32
    assert second == 32
    assert len(calls) == 1


def test_prepare_for_load_keeps_same_model(tmp_path, monkeypatch):
    pool = ModelPool()
    model = tmp_path / "same.gguf"
    model.write_bytes(b"x")
    norm = str(model.resolve())
    pool._active = LoadedModel(
        key=f"llama:{norm}",
        backend=BackendKind.LLAMA,
        handle=object(),
        meta={"path": str(model), "norm_path": norm},
    )
    refreshed = {"calls": 0}

    def _refresh(force_refresh=False):
        refreshed["calls"] += 1
        return {}

    monkeypatch.setattr("seiso.hardware.profile.hardware_profile", _refresh)
    unloaded = pool.prepare_for_load(str(model), BackendKind.LLAMA)
    assert unloaded is False
    assert pool.active_key is not None
    assert refreshed["calls"] == 0


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


def test_llama_reuses_cached_model_when_context_grows(monkeypatch, tmp_path):
    from seiso.inference import model_pool

    pool = ModelPool()
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    load_ctx: list[int] = []

    class FakeLlama:
        pass

    def fake_load(path, n_ctx):
        load_ctx.append(n_ctx)
        return FakeLlama()

    monkeypatch.setattr(model_pool, "_load_llama_model", fake_load)
    monkeypatch.setattr(
        "seiso.inference.tuning.attach_llama_prompt_cache",
        lambda _llm: None,
    )

    pool.get_llama(str(model_path), n_ctx=4096)
    pool.get_llama(str(model_path), n_ctx=8192)

    assert load_ctx == [4096, 8192]


def test_llama_reuses_larger_preloaded_context(monkeypatch, tmp_path):
    from seiso.inference import model_pool

    pool = ModelPool()
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    handle = object()
    load_paths: list[str] = []

    class FakeLlama:
        pass

    def fake_load(path, n_ctx):
        load_paths.append(path)
        return FakeLlama()

    monkeypatch.setattr(model_pool, "_load_llama_model", fake_load)
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
    monkeypatch.setattr("seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True)

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
    monkeypatch.setattr("seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True)

    kwargs = llama_load_kwargs(4096)
    assert kwargs["n_gpu_layers"] == -1
    assert kwargs["flash_attn"] is True
    assert kwargs["no_perf"] is True


def test_llama_load_kwargs_cuda_defaults(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SEISO_LLAMA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(os, "cpu_count", lambda: 24)
    monkeypatch.setattr("seiso.inference.model_pool._cuda_available", lambda: True)
    monkeypatch.setattr("seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 9000)
    monkeypatch.setattr("seiso.inference.model_pool._speed_llama_batch_defaults", lambda _h: None)

    kwargs = llama_load_kwargs(4096)
    assert kwargs["n_gpu_layers"] == -1
    assert kwargs["n_threads"] == 12
    assert kwargs["n_threads_batch"] == 16
    assert kwargs["n_batch"] == 1792
    assert kwargs["n_ubatch"] == 768
    assert kwargs["flash_attn"] is True
    assert kwargs["offload_kqv"] is True
    assert kwargs["op_offload"] is True


def test_llama_load_kwargs_threads_batch_override(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SEISO_LLAMA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    monkeypatch.setattr("seiso.inference.model_pool._llama_gpu_offload_ok", lambda: False)
    monkeypatch.setenv("SEISO_LLAMA_THREADS_BATCH", "5")

    kwargs = llama_load_kwargs(2048)

    assert kwargs["n_threads_batch"] == 5


def test_llama_load_kwargs_nvidia_smi_without_cuda_torch(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SEISO_LLAMA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr("seiso.inference.model_pool._cuda_available", lambda: False)
    monkeypatch.setattr("seiso.inference.model_pool._nvidia_hardware_visible", lambda: True)
    monkeypatch.setattr("seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 16384)

    kwargs = llama_load_kwargs(4096)
    assert kwargs["n_gpu_layers"] == -1


def test_llama_load_retryable_detects_context_and_file_errors():
    import seiso.inference.model_pool as mp

    assert mp._llama_load_retryable(ValueError("Failed to create llama_context"))
    assert mp._llama_load_retryable(ValueError("Failed to load model from file: x.gguf"))
    assert not mp._llama_load_retryable(ValueError("invalid n_ctx"))


def test_llama_layer_attempts_partial_descending(monkeypatch, tmp_path):
    import seiso.inference.model_pool as mp

    gguf = tmp_path / "big.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr(mp, "fit_llama_gpu_layers", lambda _p, _r, _h: 24)
    monkeypatch.setattr("seiso.inference.backends.gguf_block_count", lambda _p: 32)

    attempts = mp._llama_layer_attempts(str(gguf), -1, 4096)
    assert -1 not in attempts
    assert 24 in attempts
    assert attempts[-1] == 0
    assert attempts[0] >= 24


def test_llama_full_gpu_targets(monkeypatch):
    import seiso.inference.model_pool as mp

    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    assert mp._llama_full_gpu_targets(-1) == [-1]
    assert mp._llama_full_gpu_targets(12) == [12]
    assert mp._llama_full_gpu_targets(0) == []


def test_headroom_llama_batch_caps_speed_first():
    import seiso.inference.model_pool as mp

    batch, ubatch = mp._headroom_llama_batch_caps(7000, speed_first=True)
    assert batch == 1024
    assert ubatch == 768

    batch, ubatch = mp._headroom_llama_batch_caps(7000, speed_first=False)
    assert batch == 1024
    assert ubatch == 512

    batch, ubatch = mp._headroom_llama_batch_caps(14000, speed_first=True)
    assert batch == 2048
    assert ubatch == 1024


def test_llama_speed_memory_profiles_non_borderline(monkeypatch, tmp_path):
    import seiso.inference.model_pool as mp

    gguf = tmp_path / "small.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr(mp, "_llama_borderline_vram", lambda _p, _f: False)
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb",
        lambda _p: 4096,
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_block_count",
        lambda _p: 32,
    )

    profiles = mp._llama_speed_memory_profiles(
        {"n_gpu_layers": -1, "n_ctx": 4096, "n_batch": 1024, "n_ubatch": 512},
        str(gguf),
        24000,
    )
    assert profiles == [{"n_batch": 2048, "n_ubatch": 1024}]


def test_llama_speed_memory_profiles_skips_borderline(monkeypatch, tmp_path):
    import seiso.inference.model_pool as mp

    gguf = tmp_path / "big.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr(mp, "_llama_borderline_vram", lambda _p, _f: True)

    profiles = mp._llama_speed_memory_profiles(
        {"n_gpu_layers": -1, "n_ctx": 4096, "n_batch": 1024, "n_ubatch": 512},
        str(gguf),
        12000,
    )
    assert profiles == []


def test_llama_load_memory_profiles_adds_compact_for_borderline(monkeypatch, tmp_path):
    import seiso.inference.model_pool as mp

    gguf = tmp_path / "big.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr(mp, "_llama_borderline_vram", lambda _p, _f: True)

    profiles = mp._llama_load_memory_profiles(
        {"n_gpu_layers": -1, "n_batch": 2048, "n_ubatch": 512},
        8192,
        str(gguf),
        12000,
    )
    assert profiles[0] == {}
    assert profiles[1]["n_ctx"] == 2048
    assert profiles[-1]["n_batch"] == 256


def test_llama_load_kwargs_forces_zero_gpu_layers_on_cpu_only_wheel(monkeypatch):
    """Linux with NVIDIA hardware but CPU-only llama-cpp-python wheel should not
    attempt GPU offload (would crash at Llama init)."""
    import seiso.inference.model_pool as mp

    for key in list(os.environ):
        if key.startswith("SEISO_LLAMA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(mp, "_cuda_available", lambda: True)
    monkeypatch.setattr(mp, "_nvidia_hardware_visible", lambda: True)
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: False)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 16384)

    kwargs = llama_load_kwargs(4096)
    assert kwargs["n_gpu_layers"] == 0
    assert "op_offload" not in kwargs
    assert kwargs["offload_kqv"] is False


def test_llama_load_kwargs_env_override_respected_when_gpu_supported(monkeypatch):
    """SEISO_LLAMA_GPU_LAYERS env var is honored when GPU offload is supported."""
    import seiso.inference.model_pool as mp

    for key in list(os.environ):
        if key.startswith("SEISO_LLAMA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(mp, "_cuda_available", lambda: True)
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 16384)
    monkeypatch.setenv("SEISO_LLAMA_GPU_LAYERS", "10")

    kwargs = llama_load_kwargs(4096)
    assert kwargs["n_gpu_layers"] == 10


def test_llama_load_kwargs_env_override_zeroed_when_gpu_unsupported(monkeypatch):
    """SEISO_LLAMA_GPU_LAYERS=99 is forced to 0 when the wheel can't offload."""
    import seiso.inference.model_pool as mp

    for key in list(os.environ):
        if key.startswith("SEISO_LLAMA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(mp, "_cuda_available", lambda: True)
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: False)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 16384)
    monkeypatch.setenv("SEISO_LLAMA_GPU_LAYERS", "99")

    kwargs = llama_load_kwargs(4096)
    assert kwargs["n_gpu_layers"] == 0


def test_platform_profile_linux_nvidia_cpu_only_wheel(monkeypatch):
    """When llama-cpp-python lacks GPU offload, platform_profile should set
    SEISO_LLAMA_GPU_LAYERS=0 even on Linux NVIDIA hardware."""
    import os
    import platform as plat

    import seiso.inference.model_pool as mp
    from seiso.hardware.tiers import HardwareTier
    from seiso.memory.platform_profile import apply_platform_memory_profile

    for key in list(os.environ):
        if key.startswith("SEISO_LLAMA_") or key == "SEISO_MEMORY_PROFILE":
            monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(plat, "system", lambda: "Linux")
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: False)
    monkeypatch.setattr(
        "seiso.memory.platform_profile.classify_tier",
        lambda _p: HardwareTier.WORKSTATION,
    )
    monkeypatch.setattr("seiso.memory.platform_profile.vram_headroom_mb", lambda _p: 20480)
    monkeypatch.setattr(
        "seiso.memory.platform_profile.training_capabilities",
        lambda: {
            "nvidia_hardware": True,
            "gpu_count": 1,
            "vendor": "nvidia",
            "train_platform": "cpu",
            "supports_mlx_inference": False,
        },
    )

    apply_platform_memory_profile(
        profile={
            "ram_gb": 32,
            "gpus": [{"name": "NVIDIA RTX 4090", "vram_total_mb": 24576}],
            "backend": "torch",
            "platform": "Linux",
        }
    )
    assert os.environ.get("SEISO_LLAMA_GPU_LAYERS") == "0"


def test_platform_caps_bnb_unavailable_on_linux(monkeypatch):
    """training_capabilities should report supports_qlora=False when
    bitsandbytes is not importable, even on Linux with NVIDIA hardware."""
    import builtins
    import platform as plat

    from seiso.kernels.platform import GpuPlatform, GpuVendor, detect_gpu
    from seiso.training.platform_caps import training_capabilities

    training_capabilities.cache_clear()
    detect_gpu.cache_clear()
    monkeypatch.setattr(
        "seiso.training.platform_caps.detect_gpu",
        lambda: GpuPlatform(
            vendor=GpuVendor.NVIDIA,
            device_name="RTX 4090",
            device_count=1,
            supports_native_cuda=False,
            supports_triton=False,
        ),
    )
    monkeypatch.setattr(plat, "system", lambda: "Linux")

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _FakeTorch:
        cuda = _FakeCuda()
        backends = type(
            "Backends",
            (),
            {"mps": type("Mps", (), {"is_available": staticmethod(lambda: False)})()},
        )()

    import sys

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "bitsandbytes":
            raise ImportError("no bnb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    caps = training_capabilities()
    assert caps["supports_qlora"] is False
    assert caps["recommended_quant"] == "16bit"
    training_capabilities.cache_clear()
    detect_gpu.cache_clear()
