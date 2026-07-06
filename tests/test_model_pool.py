"""Tests for VRAM model pool."""

import os
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from seiso.inference.model_pool import (
    BackendKind,
    LoadedModel,
    ModelPool,
    llama_load_kwargs,
)


def _write_arch_gguf(
    path: Path, architecture: str, *, extra: list[tuple[bytes, int]] | None = None
) -> None:
    import struct

    arch_key = b"general.architecture"
    arch_value = architecture.encode()
    prefix = architecture.split("-", 1)[0]
    payload = [
        struct.pack("<Q", len(arch_key)),
        arch_key,
        struct.pack("<I", 8),
        struct.pack("<Q", len(arch_value)),
        arch_value,
    ]
    for key, value in extra or []:
        payload.extend(
            [
                struct.pack("<Q", len(key)),
                key,
                struct.pack("<I", 4),
                struct.pack("<I", value),
            ]
        )
    block_key = prefix.encode() + b".block_count"
    payload.extend(
        [
            struct.pack("<Q", len(block_key)),
            block_key,
            struct.pack("<I", 4),
            struct.pack("<I", 40),
        ]
    )
    kv_count = 2 + len(extra or [])
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, kv_count) + b"".join(payload))


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


def test_cancel_and_unload_defers_while_inference_active(tmp_path):
    pool = ModelPool()
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    pool._active = LoadedModel(
        key="llama:model",
        backend=BackendKind.LLAMA,
        handle=object(),
        meta={"path": str(model), "norm_path": str(model.resolve())},
    )
    pool.begin_inference()
    pool.cancel_and_unload()
    assert pool.active_key is not None
    pool.end_inference()
    assert pool.active_key is None


def test_switch_waits_for_inference_before_replacing(tmp_path):
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
    pool.begin_inference()

    def delayed_end() -> None:
        time.sleep(0.15)
        pool.end_inference()

    threading.Thread(target=delayed_end, daemon=True).start()
    started = time.time()
    pool.switch(str(second_path), BackendKind.LLAMA, lambda _path: second)
    elapsed = time.time() - started

    assert first.closed is True
    assert pool.status()["path"] == str(second_path.absolute())
    assert elapsed >= 0.1


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


def test_prepare_for_load_waits_for_inference(tmp_path, monkeypatch):
    pool = ModelPool()
    first = tmp_path / "a.gguf"
    second = tmp_path / "b.gguf"
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    class FakeLlama:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    handle = FakeLlama()
    pool._active = LoadedModel(
        key="llama:a",
        backend=BackendKind.LLAMA,
        handle=handle,
        meta={"path": str(first), "norm_path": str(first.resolve())},
    )
    monkeypatch.setattr("seiso.hardware.profile.hardware_profile", lambda **_: {})
    pool.begin_inference()

    def delayed_end() -> None:
        time.sleep(0.15)
        pool.end_inference()

    threading.Thread(target=delayed_end, daemon=True).start()
    started = time.time()
    unloaded = pool.prepare_for_load(str(second), BackendKind.LLAMA)
    elapsed = time.time() - started

    assert unloaded is True
    assert pool.active_key is None
    assert handle.closed is True
    assert elapsed >= 0.1


def test_reload_llama_preserves_generation_and_skips_self_wait(tmp_path, monkeypatch):
    from seiso.inference import model_pool as pool_mod

    pool = ModelPool()
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")

    class FakeLlama:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    first = FakeLlama()
    second = FakeLlama()
    pool._active = LoadedModel(
        key="llama:model",
        backend=BackendKind.LLAMA,
        handle=first,
        meta={
            "path": str(model),
            "norm_path": str(model.resolve()),
            "n_ctx": 4096,
            "load_tier": "normal",
        },
    )
    gen_id = pool.bump_generation()
    pool.begin_inference()

    def fake_load(path, n_ctx, tier="normal", batch_override=None):
        return second

    monkeypatch.setattr(pool_mod, "_load_llama_model", fake_load)
    monkeypatch.setattr(pool_mod, "_clear_optimal_layers_cache", lambda: None)
    monkeypatch.setattr(pool_mod, "_refresh_headroom_stats", lambda force=False: None)
    monkeypatch.setattr("seiso.memory.protection.ensure_load_fits", lambda *a, **k: {})
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda *a, **k: 100)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 10_000)
    monkeypatch.setattr("seiso.memory.protection.release_cached_memory", lambda **k: None)
    monkeypatch.setattr("seiso.inference.llama_vision.resolve_mmproj_path", lambda *a, **k: None)

    started = time.time()
    handle = pool.reload_llama(
        str(model),
        4096,
        tier="compact",
        batch_override=(512, 128),
    )
    elapsed = time.time() - started

    assert handle is second
    assert pool.is_generation_active(gen_id)
    assert first.closed is True
    assert elapsed < 1.0
    assert pool._inference_refs == 1


def test_get_llama_same_model_reload_preserves_generation(tmp_path, monkeypatch):
    from seiso.inference import model_pool as pool_mod

    pool = ModelPool()
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")

    class FakeLlama:
        def __init__(self, *, n_ctx: int) -> None:
            self.closed = False
            self._seiso_n_ctx = n_ctx
            self._seiso_n_gpu_layers = -1

        def close(self) -> None:
            self.closed = True

    warm = FakeLlama(n_ctx=2048)
    upgraded = FakeLlama(n_ctx=8192)
    pool._active = LoadedModel(
        key=f"llama:{model.resolve()}",
        backend=BackendKind.LLAMA,
        handle=warm,
        meta={
            "path": str(model),
            "norm_path": str(model.resolve()),
            "n_ctx": 2048,
            "load_tier": "normal",
            "n_gpu_layers": -1,
        },
    )
    gen_id = pool.bump_generation()
    pool.begin_inference()

    def fake_load(path, n_ctx, tier="normal", batch_override=None):
        return upgraded

    monkeypatch.setattr(pool_mod, "_load_llama_model", fake_load)
    monkeypatch.setattr(pool_mod, "_clear_optimal_layers_cache", lambda: None)
    monkeypatch.setattr(pool_mod, "_refresh_headroom_stats", lambda force=False: None)
    monkeypatch.setattr(pool_mod, "_llama_cache_is_optimal", lambda *a, **k: True)
    monkeypatch.setattr(pool_mod, "_llama_cache_headroom_ok", lambda *a, **k: True)
    monkeypatch.setattr("seiso.memory.protection.ensure_load_fits", lambda *a, **k: {})
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda *a, **k: 100)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 10_000)
    monkeypatch.setattr("seiso.memory.protection.release_cached_memory", lambda **k: None)
    monkeypatch.setattr("seiso.inference.llama_vision.resolve_mmproj_path", lambda *a, **k: None)

    started = time.time()
    handle = pool.get_llama(str(model), n_ctx=8192)
    elapsed = time.time() - started

    assert handle is upgraded
    assert pool.is_generation_active(gen_id)
    assert warm.closed is True
    assert elapsed < 1.0
    assert pool._inference_refs == 1


def test_torch_same_path_different_cache_key_invalidates_generation(tmp_path, monkeypatch):
    from seiso.inference import model_pool as pool_mod

    pool = ModelPool()
    target = tmp_path / "target"
    target.mkdir()

    class FakeHandle:
        pass

    old_handle = FakeHandle()
    new_handle = FakeHandle()
    norm = str(target.resolve())
    pool._active = LoadedModel(
        key=f"spec:{norm}:/tmp/draft",
        backend=BackendKind.TORCH,
        handle=old_handle,
        meta={
            "path": str(target),
            "norm_path": norm,
            "draft_path": "/tmp/draft",
        },
    )
    gen_id = pool.bump_generation()

    monkeypatch.setattr(pool_mod, "_clear_optimal_layers_cache", lambda: None)
    monkeypatch.setattr(pool_mod, "_refresh_headroom_stats", lambda force=False: None)
    monkeypatch.setattr("seiso.memory.protection.ensure_load_fits", lambda *a, **k: {})
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda *a, **k: 100)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 10_000)
    monkeypatch.setattr("seiso.memory.protection.release_cached_memory", lambda **k: None)

    handle = pool.switch(
        str(target),
        BackendKind.TORCH,
        lambda _path: new_handle,
        cache_key=f"torch:{norm}",
    )

    assert handle is new_handle
    assert not pool.is_generation_active(gen_id)
    assert pool.active_key == f"torch:{norm}"


def test_torch_speculative_low_memory_preflight_is_advisory(tmp_path, monkeypatch, caplog):
    pool = ModelPool()
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()

    monkeypatch.setattr("seiso.memory.protection.ensure_load_fits", lambda *a, **k: {})
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda *a, **k: 10_000)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 100)
    monkeypatch.setattr("seiso.memory.protection.release_cached_memory", lambda **k: None)

    def fake_load(path: str, *, load_in_4bit: bool = True):
        return f"model:{Path(path).name}", f"tokenizer:{Path(path).name}"

    monkeypatch.setattr(pool, "_load_torch_pair", fake_load)

    with caplog.at_level("WARNING"):
        bundle = pool.get_torch_speculative(str(target), str(draft))

    assert bundle.target_model == "model:target"
    assert bundle.draft_model == "model:draft"
    assert "Speculative pair may exceed free memory" in caplog.text


def test_llama_gpu_layers_optimal_uses_short_ttl_cache(monkeypatch, tmp_path):
    import seiso.inference.model_pool as mp

    mp._clear_optimal_layers_cache()
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"gguf")
    calls: list[int] = []

    monkeypatch.setattr(
        mp,
        "fit_llama_gpu_layers",
        lambda _p, _r, _h, **kwargs: calls.append(kwargs.get("n_ctx", 0)) or 32,
    )

    first = mp._llama_gpu_layers_optimal(str(gguf), -1, n_ctx=4096)
    second = mp._llama_gpu_layers_optimal(str(gguf), -1, n_ctx=4096)
    third = mp._llama_gpu_layers_optimal(str(gguf), -1, n_ctx=8192)

    assert first == 32
    assert second == 32
    assert third == 32
    assert calls == [4096, 8192]


def test_native_linux_full_offload_cache_invalidates_when_partial_is_now_optimal(
    monkeypatch, tmp_path
):
    import seiso.inference.model_pool as mp

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"gguf")
    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(mp, "_llama_gpu_layers_optimal", lambda *_a, **_k: 32)

    assert not mp._llama_cache_is_optimal(str(gguf), -1, -1, n_ctx=4096)


def test_non_native_full_offload_cache_stays_optimal(monkeypatch, tmp_path):
    import seiso.inference.model_pool as mp

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"gguf")
    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: False)

    assert mp._llama_cache_is_optimal(str(gguf), -1, -1, n_ctx=4096)


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


def test_would_switch_model_spec_bundle_same_torch_path(tmp_path):
    pool = ModelPool()
    target = tmp_path / "target"
    target.mkdir()
    norm = str(target.resolve())
    pool._active = LoadedModel(
        key=f"spec:{norm}:{norm}-draft",
        backend=BackendKind.TORCH,
        handle=object(),
        meta={
            "path": str(target),
            "norm_path": norm,
            "draft_path": str(tmp_path / "draft"),
        },
    )

    assert pool.would_switch_model(str(target), BackendKind.TORCH) is True


def test_prepare_for_load_unloads_spec_bundle_for_same_torch_path(tmp_path, monkeypatch):
    pool = ModelPool()
    target = tmp_path / "target"
    target.mkdir()
    norm = str(target.resolve())
    pool._active = LoadedModel(
        key=f"spec:{norm}:{norm}-draft",
        backend=BackendKind.TORCH,
        handle=object(),
        meta={
            "path": str(target),
            "norm_path": norm,
            "draft_path": str(tmp_path / "draft"),
        },
    )
    monkeypatch.setattr("seiso.hardware.profile.hardware_profile", lambda **_: {})

    unloaded = pool.prepare_for_load(str(target), BackendKind.TORCH)

    assert unloaded is True
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


def test_llama_reuses_cached_model_when_context_grows(monkeypatch, tmp_path):
    from seiso.inference import model_pool

    pool = ModelPool()
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    load_ctx: list[int] = []

    class FakeLlama:
        pass

    def fake_load(path, n_ctx, **_kwargs):
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


def test_get_llama_uses_actual_loaded_context_for_cache(monkeypatch, tmp_path):
    from seiso.inference import model_pool

    pool = ModelPool()
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    load_ctx: list[int] = []

    class FakeLlama:
        def __init__(self, actual_ctx: int) -> None:
            self._seiso_n_ctx = actual_ctx
            self._seiso_n_gpu_layers = -1
            self._seiso_load_headroom_mb = 24576

    def fake_load(_path, n_ctx, **_kwargs):
        load_ctx.append(n_ctx)
        # Simulate a successful fallback profile that lowered n_ctx to survive OOM.
        return FakeLlama(2048 if n_ctx == 8192 else n_ctx)

    monkeypatch.setattr(model_pool, "_load_llama_model", fake_load)
    monkeypatch.setattr(model_pool, "_llama_cache_is_optimal", lambda *_a, **_k: True)
    monkeypatch.setattr(model_pool, "_llama_cache_headroom_ok", lambda _h: True)

    pool.get_llama(str(model_path), n_ctx=8192)
    pool.get_llama(str(model_path), n_ctx=4096)

    assert load_ctx == [8192, 4096]


def test_get_llama_unloads_previous_handle_when_reloading(monkeypatch, tmp_path):
    from seiso.inference import model_pool

    pool = ModelPool()
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    handles: list[object] = []

    class FakeLlama:
        def __init__(self, n_ctx: int) -> None:
            self.n_ctx = n_ctx
            self.closed = False

        def close(self) -> None:
            self.closed = True

    def fake_load(_path, n_ctx, **_kwargs):
        llm = FakeLlama(n_ctx)
        handles.append(llm)
        return llm

    monkeypatch.setattr(model_pool, "_load_llama_model", fake_load)
    monkeypatch.setattr(
        "seiso.inference.tuning.attach_llama_prompt_cache",
        lambda _llm: None,
    )

    pool.get_llama(str(model_path), n_ctx=4096)
    pool.get_llama(str(model_path), n_ctx=8192)

    assert len(handles) == 2
    assert handles[0].closed is True


def test_get_llama_reloads_when_cached_headroom_stale(monkeypatch, tmp_path):
    from seiso.inference import model_pool

    pool = ModelPool()
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    load_count = {"n": 0}

    class FakeLlama:
        _seiso_n_gpu_layers = -1
        _seiso_load_headroom_mb = 24576

    def fake_load(_path, n_ctx, **_kwargs):
        load_count["n"] += 1
        return FakeLlama()

    monkeypatch.setattr(model_pool, "_load_llama_model", fake_load)
    monkeypatch.setattr(model_pool, "_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(
        "seiso.inference.tuning.attach_llama_prompt_cache",
        lambda _llm: None,
    )
    monkeypatch.setattr(model_pool, "_llama_cache_is_optimal", lambda *_a, **_k: True)
    headroom_ok = {"value": True}
    monkeypatch.setattr(
        model_pool,
        "_llama_cache_headroom_ok",
        lambda _handle: headroom_ok["value"],
    )

    pool.get_llama(str(model_path), n_ctx=4096)
    headroom_ok["value"] = False
    pool.get_llama(str(model_path), n_ctx=4096)

    assert load_count["n"] == 2


def test_llama_reuses_larger_preloaded_context(monkeypatch, tmp_path):
    from seiso.inference import model_pool

    pool = ModelPool()
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    handle = object()
    load_paths: list[str] = []

    class FakeLlama:
        pass

    def fake_load(path, n_ctx, **_kwargs):
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


def test_dflash_loader_reuses_vram_aware_llama_loader(monkeypatch, tmp_path):
    from seiso.inference import model_pool

    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"gguf")
    calls: list[tuple[str, int]] = []
    handle = object()

    def fake_load(path, n_ctx):
        calls.append((path, n_ctx))
        return handle

    monkeypatch.setattr(model_pool, "_load_llama_model", fake_load)

    assert model_pool._load_dflash_llm(str(draft), 2048) is handle
    assert calls == [(str(draft), 2048)]


def test_dflash_cache_reloads_when_larger_context_is_needed(monkeypatch, tmp_path):
    from seiso.inference import model_pool

    model_pool.clear_dflash_draft_cache()
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"gguf")
    handles: list[object] = []
    closed: list[object] = []

    class FakeLlama:
        def close(self) -> None:
            closed.append(self)

    def fake_load(_path, _n_ctx):
        handle = FakeLlama()
        handles.append(handle)
        return handle

    monkeypatch.setattr(
        "seiso.inference.backends.prepare_model_path",
        lambda path, _backend: str(path),
    )
    monkeypatch.setattr(model_pool, "_load_dflash_llm", fake_load)

    first = model_pool.get_dflash_draft(str(draft), n_ctx=2048)
    same = model_pool.get_dflash_draft(str(draft), n_ctx=1024)
    first_llm = first.llm
    larger = model_pool.get_dflash_draft(str(draft), n_ctx=4096)

    assert first is same
    assert larger is not first
    assert first.n_ctx == 2048
    assert larger.n_ctx == 4096
    assert handles == [first_llm, larger.llm]
    assert closed == [first_llm]
    assert first.llm is None  # disposed under lock
    model_pool.clear_dflash_draft_cache()


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
    for key in list(os.environ):
        if key.startswith("SEISO_LLAMA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr("seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr("seiso.inference.model_pool._native_linux_nvidia", lambda: False)

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
    monkeypatch.setattr(
        "seiso.inference.model_pool._available_cpu_count", lambda: 24
    )
    monkeypatch.setattr("seiso.inference.model_pool._cuda_available", lambda: True)
    monkeypatch.setattr("seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr("seiso.inference.model_pool._native_linux_nvidia", lambda: False)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)

    kwargs = llama_load_kwargs(4096)
    assert kwargs["n_gpu_layers"] == -1
    assert kwargs["n_threads"] == 16
    assert kwargs["n_threads_batch"] == 24
    assert kwargs["n_batch"] == 4096
    assert kwargs["n_ubatch"] == 1024
    assert kwargs["flash_attn"] is True
    assert kwargs["offload_kqv"] is True
    assert kwargs["op_offload"] is True


def test_llama_load_kwargs_native_linux_nvidia_defaults(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SEISO_LLAMA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr("seiso.inference.model_pool._cuda_available", lambda: True)
    monkeypatch.setattr("seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr("seiso.inference.model_pool._native_linux_nvidia", lambda: True)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 1024)
    monkeypatch.setattr("seiso.inference.backends.gguf_block_count", lambda _p: 32)
    monkeypatch.setattr("seiso.memory.protection.discrete_gpu_total_mb", lambda _p=None: 24576)

    kwargs = llama_load_kwargs(4096, model_path="/tmp/model.gguf")
    from seiso.memory.protection import gpu_batch_tier_caps

    expected_batch, expected_ubatch = gpu_batch_tier_caps(24576, "normal")
    assert kwargs["n_batch"] == expected_batch
    assert kwargs["n_ubatch"] == expected_ubatch
    assert "flash_attn" not in kwargs


def test_native_linux_flash_attn_opt_in(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SEISO_LLAMA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr("seiso.inference.model_pool._cuda_available", lambda: True)
    monkeypatch.setattr("seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr("seiso.inference.model_pool._native_linux_nvidia", lambda: True)
    monkeypatch.setenv("SEISO_LLAMA_FLASH_ATTN", "true")

    kwargs = llama_load_kwargs(4096, model_path="/tmp/model.gguf")
    assert kwargs["flash_attn"] is True


def test_native_linux_flash_attn_defaults_off_dense_opt_in(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("SEISO_LLAMA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr("seiso.inference.model_pool._cuda_available", lambda: True)
    monkeypatch.setattr("seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr("seiso.inference.model_pool._native_linux_nvidia", lambda: True)
    monkeypatch.setattr("seiso.inference.model_pool._llama_skip_partial_offload", lambda _p: False)

    kwargs = llama_load_kwargs(4096, model_path="/tmp/model.gguf")
    assert "flash_attn" not in kwargs

    monkeypatch.setenv("SEISO_LLAMA_FLASH_ATTN", "true")
    kwargs = llama_load_kwargs(4096, model_path="/tmp/model.gguf")
    assert kwargs.get("flash_attn") is True


def test_llama_gpu_offload_ok_retries_after_import_failure(monkeypatch):
    import builtins
    import sys

    import seiso.inference.model_pool as mp

    mp.reset_llama_gpu_offload_cache()
    attempts = {"n": 0}
    orig_import = builtins.__import__

    class FakeLlamaCpp:
        @staticmethod
        def llama_supports_gpu_offload():
            return True

    def fake_import(name, *args, **kwargs):
        if name == "llama_cpp":
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ImportError("libcudart.so.12: cannot open shared object file")
            return sys.modules["llama_cpp"]
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr("seiso.platform.ensure_cuda_library_path", lambda: [])
    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert mp._llama_gpu_offload_ok() is False
    assert mp._llama_offload_checked is False

    monkeypatch.setitem(sys.modules, "llama_cpp", FakeLlamaCpp())
    assert mp._llama_gpu_offload_ok() is True
    assert mp._llama_offload_checked is True
    assert attempts["n"] == 2


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
    assert mp._llama_load_retryable(RuntimeError("failed to allocate Metal buffer"))
    assert not mp._llama_load_retryable(ValueError("invalid n_ctx"))


def test_llama_layer_attempts_partial_descending(monkeypatch, tmp_path):
    import seiso.inference.model_pool as mp

    gguf = tmp_path / "big.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr(mp, "fit_llama_gpu_layers", lambda _p, _r, _h, **_k: 24)
    monkeypatch.setattr("seiso.inference.backends.gguf_block_count", lambda _p: 32)

    attempts = mp._llama_layer_attempts(str(gguf), -1, 4096)
    assert -1 not in attempts
    assert 24 in attempts
    assert attempts[-1] == 0
    assert attempts[0] >= 24


def test_llama_layer_attempts_mac_cpu_offload_ladder(monkeypatch, tmp_path):
    import seiso.inference.model_pool as mp

    gguf = tmp_path / "apple-big.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr("seiso.inference.backends.gguf_block_count", lambda _p: 40)

    attempts = mp._llama_layer_attempts(str(gguf), -1, 32768)

    assert attempts[:3] == [39, 35, 30]
    assert 20 in attempts
    assert attempts[-1] == 0


def test_llama_layer_attempts_mac_cpu_offload_can_be_disabled(monkeypatch, tmp_path):
    import seiso.inference.model_pool as mp

    gguf = tmp_path / "apple-big.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setenv("SEISO_LLAMA_MAC_CPU_OFFLOAD", "0")
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr(mp, "fit_llama_gpu_layers", lambda _p, _r, _h, **_k: 16)
    monkeypatch.setattr("seiso.inference.backends.gguf_block_count", lambda _p: 40)

    attempts = mp._llama_layer_attempts(str(gguf), -1, 32768)

    assert attempts[0] >= 16
    assert attempts[-1] == 0
    assert 16 in attempts


def test_llama_partial_profiles_mac_preserve_fast_attempts(monkeypatch):
    import seiso.inference.model_pool as mp

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    profiles = [{"n_batch": 4096}, {}, {"n_batch": 256}]

    assert mp._llama_partial_memory_profiles(profiles) == profiles
    assert mp._llama_partial_kqv_options() == [{}, {"offload_kqv": False}]


def test_llama_partial_profiles_non_mac_use_lean_fallback(monkeypatch):
    import seiso.inference.model_pool as mp

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    profiles = [{"n_batch": 4096}, {}, {"n_batch": 256}]

    assert mp._llama_partial_memory_profiles(profiles) == [{"n_batch": 256}]
    assert mp._llama_partial_kqv_options() == [{}]


def test_llama_full_gpu_targets(monkeypatch):
    import seiso.inference.model_pool as mp

    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    assert mp._llama_full_gpu_targets(-1) == [-1]
    assert mp._llama_full_gpu_targets(12) == [12]
    assert mp._llama_full_gpu_targets(0) == []


def test_llama_batch_defaults_are_speed_first(monkeypatch):
    import seiso.inference.model_pool as mp

    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: False)
    batch, ubatch = mp._llama_batch_defaults()
    assert batch == 4096
    assert ubatch == 1024


def test_llama_batch_defaults_match_july3_speed_first(monkeypatch):
    import seiso.inference.model_pool as mp
    from seiso.memory.protection import gpu_batch_tier_caps

    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(
        "seiso.memory.protection.discrete_gpu_total_mb",
        lambda _profile=None: 24576,
    )
    batch, ubatch = mp._llama_batch_defaults()
    expected_batch, expected_ubatch = gpu_batch_tier_caps(24576, "normal")
    assert batch == expected_batch
    assert ubatch == expected_ubatch


def test_llama_load_model_tries_speed_profile_before_base(monkeypatch, tmp_path):
    import seiso.inference.model_pool as mp

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"gguf")
    attempts: list[tuple[int, int]] = []

    class FakeLlama:
        def __init__(self, *, model_path: str, **kwargs):
            assert model_path == str(gguf)
            attempts.append((kwargs["n_batch"], kwargs["n_ubatch"]))
            self._seiso_n_gpu_layers = kwargs["n_gpu_layers"]

    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr(mp, "_default_llama_gpu_layers", lambda: -1)
    monkeypatch.setattr(mp, "_llama_kv_quant_options", lambda _p: [{}])
    monkeypatch.setattr(
        "seiso.memory.protection.llama_load_profile_ladder",
        lambda **_kwargs: [
            {"n_batch": 4096, "n_ubatch": 1024},
            {"n_batch": 512, "n_ubatch": 256},
        ],
    )
    monkeypatch.setattr(mp, "_llama_full_gpu_targets", lambda _r: [-1])
    monkeypatch.setattr(mp, "_llama_layer_attempts", lambda *_a, **_k: [0])
    monkeypatch.setattr(mp, "_refresh_headroom_stats", lambda *, force=False: None)
    monkeypatch.setattr("seiso.memory.protection.release_cached_memory", lambda sync=False: None)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 1024)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24000)
    monkeypatch.setattr(
        "seiso.inference.tuning.attach_llama_prompt_cache",
        lambda _llm, **_: None,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "llama_cpp",
        type("LlamaCpp", (), {"Llama": FakeLlama}),
    )

    mp._load_llama_model(str(gguf), 4096)

    assert attempts[0][0] <= 4096
    assert attempts[0][1] <= 1024


def test_native_linux_load_model_uses_crash_resistant_kwargs(monkeypatch, tmp_path):
    import seiso.inference.model_pool as mp

    gguf = tmp_path / "qwen-27b-q4.gguf"
    gguf.write_bytes(b"gguf")
    attempts: list[dict[str, object]] = []

    class FakeLlama:
        def __init__(self, *, model_path: str, **kwargs):
            assert model_path == str(gguf)
            attempts.append(dict(kwargs))
            self._seiso_n_gpu_layers = kwargs["n_gpu_layers"]

    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr(mp, "_default_llama_gpu_layers", lambda: -1)
    monkeypatch.setattr(mp, "_llama_kv_quant_options", lambda _p: [{}])
    monkeypatch.setattr(mp, "_llama_full_gpu_targets", lambda _r: [-1])
    monkeypatch.setattr(mp, "_llama_layer_attempts", lambda *_a, **_k: [24, 0])
    monkeypatch.setattr(mp, "gguf_total_layers", lambda _p: 64)
    monkeypatch.setattr(mp, "_refresh_headroom_stats", lambda *, force=False: None)
    monkeypatch.setattr("seiso.memory.protection.release_cached_memory", lambda sync=False: None)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 17000)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr(
        "seiso.inference.tuning.attach_llama_prompt_cache",
        lambda _llm, **_: None,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "llama_cpp",
        type("LlamaCpp", (), {"Llama": FakeLlama}),
    )

    llm = mp._load_llama_model(str(gguf), 4096)

    first = attempts[0]
    assert first["n_batch"] <= 512
    assert first["n_ubatch"] <= 128
    assert first["n_gpu_layers"] == -1
    assert first["offload_kqv"] is False
    assert first["op_offload"] is False
    assert "flash_attn" not in first
    assert llm._seiso_n_batch == first["n_batch"]
    assert llm._seiso_n_ubatch == first["n_ubatch"]
    assert llm._seiso_n_ctx == first["n_ctx"]
    assert llm._seiso_model_path == str(gguf)
    assert llm._seiso_load_headroom_mb == 24576


def test_native_linux_partial_offload_disables_kqv_and_op_offload(monkeypatch, tmp_path):
    import seiso.inference.model_pool as mp

    gguf = tmp_path / "qwen-14b-q4.gguf"
    gguf.write_bytes(b"gguf")
    attempts: list[dict[str, object]] = []

    class FakeLlama:
        def __init__(self, *, model_path: str, **kwargs):
            assert model_path == str(gguf)
            attempts.append(dict(kwargs))
            self._seiso_n_gpu_layers = kwargs["n_gpu_layers"]

    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
    monkeypatch.setattr("seiso.platform.use_linux_nvidia_inference_guards", lambda **_: True)
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr(mp, "_default_llama_gpu_layers", lambda: -1)
    monkeypatch.setattr(mp, "fit_llama_gpu_layers", lambda *_a, **_k: 24)
    monkeypatch.setattr(mp, "_llama_full_gpu_targets", lambda _r: [])
    monkeypatch.setattr(mp, "_llama_layer_attempts", lambda *_a, **_k: [24])
    monkeypatch.setattr(mp, "_llama_kv_quant_options", lambda _p: [{}])
    monkeypatch.setattr(mp, "gguf_total_layers", lambda _p: 64)
    monkeypatch.setattr(mp, "_refresh_headroom_stats", lambda *, force=False: None)
    monkeypatch.setattr(
        "seiso.memory.protection.llama_load_profile_ladder",
        lambda **_kwargs: [{"n_batch": 1024, "n_ubatch": 256}],
    )
    monkeypatch.setattr(
        "seiso.memory.protection.llama_model_is_tight_vram_fit",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr("seiso.memory.protection.release_cached_memory", lambda sync=False: None)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 9000)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr(
        "seiso.inference.tuning.attach_llama_prompt_cache",
        lambda _llm, **_: None,
    )
    monkeypatch.setattr(
        "seiso.inference.llama_vision.apply_llama_vision_load_kwargs",
        lambda kwargs, _path: kwargs,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "llama_cpp",
        type("LlamaCpp", (), {"Llama": FakeLlama}),
    )

    mp._load_llama_model(str(gguf), 4096)

    assert attempts[0]["n_gpu_layers"] == 24
    assert attempts[0]["offload_kqv"] is False
    assert attempts[0]["op_offload"] is False
    assert "flash_attn" not in attempts[0]


def test_qwen36_27b_native_linux_falls_back_to_partial_when_full_offload_fails(monkeypatch, tmp_path):
    import seiso.inference.model_pool as mp

    gguf = tmp_path / "Qwen3.6-27B-UD-Q4_K_XL.gguf"
    _write_arch_gguf(gguf, "qwen3")
    attempts: list[dict[str, object]] = []

    class FakeLlama:
        def __init__(self, *, model_path: str, **kwargs):
            assert model_path == str(gguf)
            if kwargs["n_gpu_layers"] == -1:
                raise RuntimeError("Failed to create llama_context")
            attempts.append(dict(kwargs))
            self._seiso_n_gpu_layers = kwargs["n_gpu_layers"]

    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
    monkeypatch.setattr("seiso.platform.use_linux_nvidia_inference_guards", lambda **_: True)
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr(mp, "_default_llama_gpu_layers", lambda: -1)
    monkeypatch.setattr(mp, "fit_llama_gpu_layers", lambda *_a, **_k: 48)
    monkeypatch.setattr(mp, "_llama_kv_quant_options", lambda _p: [{}])
    monkeypatch.setattr(mp, "gguf_total_layers", lambda _p: 64)
    monkeypatch.setattr(mp, "_refresh_headroom_stats", lambda *, force=False: None)
    monkeypatch.setattr("seiso.memory.protection.release_cached_memory", lambda sync=False: None)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 17_000)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24_576)
    monkeypatch.setattr("seiso.memory.protection.available_ram_mb", lambda: 65_536)
    monkeypatch.setattr(
        "seiso.memory.protection.llama_kv_cache_reserve_mb",
        lambda *_args, **_kwargs: 1024,
    )
    monkeypatch.setattr(
        "seiso.inference.tuning.attach_llama_prompt_cache",
        lambda _llm, **_: None,
    )
    monkeypatch.setattr(
        "seiso.inference.llama_vision.apply_llama_vision_load_kwargs",
        lambda kwargs, _path: kwargs,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "llama_cpp",
        type("LlamaCpp", (), {"Llama": FakeLlama}),
    )

    llm = mp._load_llama_model(str(gguf), 4096)

    assert attempts
    first = attempts[0]
    assert first["n_gpu_layers"] == 48
    assert first["offload_kqv"] is False
    assert first["op_offload"] is False
    assert "flash_attn" not in first
    assert llm._seiso_n_gpu_layers == 48


def test_qwen36_27b_native_linux_full_offloads_when_it_fits(monkeypatch, tmp_path):
    import seiso.inference.model_pool as mp

    gguf = tmp_path / "Qwen3.6-27B-UD-Q4_K_XL.gguf"
    _write_arch_gguf(gguf, "qwen3")
    attempts: list[dict[str, object]] = []

    class FakeLlama:
        def __init__(self, *, model_path: str, **kwargs):
            assert model_path == str(gguf)
            attempts.append(dict(kwargs))
            self._seiso_n_gpu_layers = kwargs["n_gpu_layers"]

    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
    monkeypatch.setattr("seiso.platform.use_linux_nvidia_inference_guards", lambda **_: True)
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr(mp, "_default_llama_gpu_layers", lambda: -1)
    monkeypatch.setattr(mp, "fit_llama_gpu_layers", lambda *_a, **_k: -1)
    monkeypatch.setattr(mp, "_llama_kv_quant_options", lambda _p: [{}])
    monkeypatch.setattr(mp, "gguf_total_layers", lambda _p: 64)
    monkeypatch.setattr(mp, "_refresh_headroom_stats", lambda *, force=False: None)
    monkeypatch.setattr("seiso.memory.protection.release_cached_memory", lambda sync=False: None)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 17_000)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24_576)
    monkeypatch.setattr("seiso.memory.protection.available_ram_mb", lambda: 65_536)
    monkeypatch.setattr(
        "seiso.memory.protection.llama_kv_cache_reserve_mb",
        lambda *_args, **_kwargs: 1024,
    )
    monkeypatch.setattr(
        "seiso.inference.tuning.attach_llama_prompt_cache",
        lambda _llm, **_: None,
    )
    monkeypatch.setattr(
        "seiso.inference.llama_vision.apply_llama_vision_load_kwargs",
        lambda kwargs, _path: kwargs,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "llama_cpp",
        type("LlamaCpp", (), {"Llama": FakeLlama}),
    )

    llm = mp._load_llama_model(str(gguf), 4096)

    assert attempts
    first = attempts[0]
    assert first["n_gpu_layers"] == -1
    assert first["n_batch"] <= 256
    assert first["n_ubatch"] <= 128
    assert "flash_attn" not in first
    assert first.get("offload_kqv") is False
    assert llm._seiso_n_gpu_layers == -1


def test_qwen3_14b_24gb_load_uses_full_gpu_kwargs(monkeypatch, tmp_path):
    import seiso.inference.model_pool as mp

    gguf = tmp_path / "qwen3-14b-q4.gguf"
    _write_arch_gguf(gguf, "qwen3")
    attempts: list[dict[str, object]] = []

    class FakeLlama:
        def __init__(self, *, model_path: str, **kwargs):
            assert model_path == str(gguf)
            attempts.append(dict(kwargs))
            self._seiso_n_gpu_layers = kwargs["n_gpu_layers"]

    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
    monkeypatch.setattr("seiso.platform.use_linux_nvidia_inference_guards", lambda **_: True)
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr(mp, "_default_llama_gpu_layers", lambda: -1)
    monkeypatch.setattr(mp, "_llama_kv_quant_options", lambda _p: [{}])
    monkeypatch.setattr(mp, "fit_llama_gpu_layers", lambda *_a, **_k: -1)
    monkeypatch.setattr(mp, "_refresh_headroom_stats", lambda *, force=False: None)
    monkeypatch.setattr("seiso.memory.protection.release_cached_memory", lambda sync=False: None)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 9000)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr("seiso.memory.protection.available_ram_mb", lambda: 65536)
    monkeypatch.setattr(
        "seiso.memory.protection.llama_kv_cache_reserve_mb",
        lambda *_args, **_kwargs: 512,
    )
    monkeypatch.setattr(
        "seiso.inference.tuning.attach_llama_prompt_cache",
        lambda _llm, **_: None,
    )
    monkeypatch.setattr(
        "seiso.inference.llama_vision.apply_llama_vision_load_kwargs",
        lambda kwargs, _path: kwargs,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "llama_cpp",
        type("LlamaCpp", (), {"Llama": FakeLlama}),
    )

    mp._load_llama_model(str(gguf), 4096)

    assert attempts
    first = attempts[0]
    assert first["n_gpu_layers"] == -1
    assert first["n_batch"] >= 512
    assert first["n_ubatch"] >= 128
    assert "flash_attn" not in first


def test_load_llama_model_records_last_safe_batch_from_override(monkeypatch, tmp_path):
    import seiso.inference.model_pool as mp

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"gguf")

    class FakeLlama:
        def __init__(self, *, model_path: str, **kwargs):
            assert model_path == str(gguf)
            self._seiso_n_gpu_layers = kwargs["n_gpu_layers"]

    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr(mp, "_default_llama_gpu_layers", lambda: 0)
    monkeypatch.setattr(mp, "_llama_kv_quant_options", lambda _p: [{}])
    monkeypatch.setattr(
        "seiso.memory.protection.llama_load_profile_ladder",
        lambda **_kwargs: [{"n_batch": 512, "n_ubatch": 128}],
    )
    monkeypatch.setattr(mp, "_llama_full_gpu_targets", lambda _r: [])
    monkeypatch.setattr(mp, "_llama_layer_attempts", lambda *_a, **_k: [0])
    monkeypatch.setattr(mp, "_refresh_headroom_stats", lambda *, force=False: None)
    monkeypatch.setattr("seiso.memory.protection.release_cached_memory", lambda sync=False: None)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 1024)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 8192)
    monkeypatch.setattr(
        "seiso.inference.tuning.attach_llama_prompt_cache",
        lambda _llm, **_: None,
    )
    monkeypatch.setattr(
        "seiso.inference.llama_vision.apply_llama_vision_load_kwargs",
        lambda kwargs, _path: kwargs,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "llama_cpp",
        type("LlamaCpp", (), {"Llama": FakeLlama}),
    )

    llm = mp._load_llama_model(str(gguf), 2048, batch_override=(512, 128))

    assert llm._seiso_last_safe_batch == 512
    assert llm._seiso_last_safe_ubatch == 128


def test_llama_load_model_tries_full_offload_before_partial_fallback(monkeypatch, tmp_path):
    import seiso.inference.model_pool as mp

    gguf = tmp_path / "qwen-27b-q4.gguf"
    gguf.write_bytes(b"gguf")
    layers_attempted: list[int] = []

    class FakeLlama:
        def __init__(self, *, model_path: str, **kwargs):
            assert model_path == str(gguf)
            if kwargs["n_gpu_layers"] == -1:
                raise RuntimeError("Failed to create llama_context")
            layers_attempted.append(kwargs["n_gpu_layers"])
            self._seiso_n_gpu_layers = kwargs["n_gpu_layers"]

    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr(mp, "_default_llama_gpu_layers", lambda: -1)
    monkeypatch.setattr(mp, "fit_llama_gpu_layers", lambda _p, _r, _h, **_k: 30)
    monkeypatch.setattr(mp, "_llama_kv_quant_options", lambda _p: [{}])
    monkeypatch.setattr(mp, "_refresh_headroom_stats", lambda *, force=False: None)
    monkeypatch.setattr(
        "seiso.memory.protection.llama_load_profile_ladder",
        lambda **_kwargs: [{"n_batch": 256, "n_ubatch": 128}],
    )
    monkeypatch.setattr("seiso.memory.protection.release_cached_memory", lambda sync=False: None)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 22000)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr("seiso.inference.backends.gguf_block_count", lambda _p: 64)
    monkeypatch.setattr(
        "seiso.inference.tuning.attach_llama_prompt_cache",
        lambda _llm, **_: None,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "llama_cpp",
        type("LlamaCpp", (), {"Llama": FakeLlama}),
    )

    mp._load_llama_model(str(gguf), 4096)

    assert layers_attempted
    assert layers_attempted[0] >= 30


def test_llama_load_profile_ladder_compact_tier(tmp_path):
    from seiso.memory.protection import (
        discrete_gpu_total_mb,
        gpu_batch_tier_caps,
        llama_load_profile_ladder,
    )

    gguf = tmp_path / "big.gguf"
    gguf.write_bytes(b"\x00" * 1024)

    profiles = llama_load_profile_ladder(
        model_path=str(gguf),
        n_ctx=8192,
        n_gpu_layers=-1,
        free_mb=12000,
        base_batch=2048,
        base_ubatch=512,
        tier="compact",
    )
    gpu_total = discrete_gpu_total_mb() or 8192
    compact_batch, _ = gpu_batch_tier_caps(gpu_total, "compact")
    assert profiles[0]["n_batch"] <= compact_batch
    assert profiles[0].get("_seiso_prompt_cache") is False


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


def test_fit_llama_gpu_layers_skips_partial_for_swa_on_linux(monkeypatch, tmp_path):
    import seiso.inference.model_pool as mp

    gguf = tmp_path / "gemma.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(mp, "_apple_silicon_metal", lambda: False)
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_uses_sliding_window_attention",
        lambda _p: True,
    )
    monkeypatch.setattr("seiso.inference.backends.gguf_total_layers", lambda _p: 42)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 9000)
    monkeypatch.setattr(
        "seiso.memory.protection.llama_offload_fits_headroom",
        lambda _path, **k: k.get("n_gpu_layers") == 27,
    )
    monkeypatch.setattr(
        "seiso.memory.protection.llama_kv_cache_reserve_mb",
        lambda *_a, **_k: 512,
    )
    monkeypatch.setattr(
        "seiso.memory.protection.llama_model_is_tight_vram_fit",
        lambda **_k: False,
    )

    layers = mp.fit_llama_gpu_layers(str(gguf), -1, 12000, n_ctx=4096)

    assert layers == 0


def test_llama_layer_attempts_cpu_only_for_swa_on_linux(monkeypatch, tmp_path):
    import seiso.inference.model_pool as mp

    gguf = tmp_path / "gemma.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(mp, "_apple_silicon_metal", lambda: False)
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_uses_sliding_window_attention",
        lambda _p: True,
    )
    monkeypatch.setattr("seiso.inference.backends.gguf_total_layers", lambda _p: 42)

    attempts = mp._llama_layer_attempts(str(gguf), -1, 12000, n_ctx=4096, fitted=24)

    assert attempts == [0]
