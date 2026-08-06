"""Bare-Forge lean footprint: unload honesty, MLX reclaim, Chat preload opt-in."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path


def test_release_cached_memory_clears_mlx_even_when_skip_probe(monkeypatch):
    from seiso.memory.protection import oom

    cleared: list[str] = []
    fake_metal = types.SimpleNamespace(clear_cache=lambda: cleared.append("metal"))
    fake_core = types.SimpleNamespace(metal=fake_metal)
    monkeypatch.setenv("SEISO_SKIP_MLX_PROBE", "1")
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)

    oom.release_cached_memory(sync=False)

    assert cleared == ["metal"]


def test_release_cached_memory_skips_torch_import_when_absent(monkeypatch):
    from seiso.memory.protection import oom

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    # Must not raise or import torch just to free caches.
    oom.release_cached_memory(sync=False)
    assert "torch" not in sys.modules


def test_cancel_and_unload_calls_unload_all_after_idle_wait(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    calls: list[str] = []

    class FakePool:
        active_inference_refs = 0

        def cancel_and_unload(self):
            calls.append("cancel")

        def _wait_for_inference_idle(self, timeout_s=30.0):
            calls.append(f"wait:{timeout_s}")
            return True

        def unload_all(self):
            calls.append("unload_all")

        def status(self):
            return {"active_model": None, "path": None}

    runner = LocalInferenceRunner.__new__(LocalInferenceRunner)
    runner._pool = FakePool()

    status = asyncio.run(runner.cancel_and_unload())

    assert calls[0] == "cancel"
    assert calls[1].startswith("wait:")
    assert calls[2] == "unload_all"
    assert status["unload_complete"] is True
    assert status["inference_idle"] is True


def test_unload_all_releases_orphan_sidecars_when_empty(monkeypatch):
    from seiso.inference.model_pool.pool import ModelPool

    notes_out: list[str] = []
    monkeypatch.setattr(
        "seiso.inference.llamaswap.release_orphan_sidecar_memory",
        lambda: ["Released Ollama resident models (keep_alive=0)"],
    )
    ModelPool.reset_instance(timeout_s=0.1)
    pool = ModelPool.get()
    pool.unload_all()
    notes_out.extend(pool.drain_release_notes())
    assert "Released Ollama resident models (keep_alive=0)" in notes_out


def test_release_external_stops_managed_vllm(monkeypatch, tmp_path: Path):
    from forge.services import memory_release

    calls: list[str] = []
    monkeypatch.setattr(memory_release, "_release_orphan_sidecars", lambda **_k: [])
    monkeypatch.setattr(
        "forge.config.get_settings",
        lambda: types.SimpleNamespace(data_dir=tmp_path),
    )
    monkeypatch.setattr(
        "forge.services.managed_vllm.stop_managed_if_running",
        lambda **_k: calls.append("stop") or {"stopped": True},
    )
    monkeypatch.setattr(
        "seiso.memory.protection.release_cached_memory",
        lambda sync=False: calls.append(f"cache:{sync}"),
    )
    monkeypatch.setattr(
        memory_release, "_refresh_hardware_profile", lambda: calls.append("refresh")
    )

    result = memory_release.release_external_inference_memory(reason="free_memory")

    assert result["managed_vllm_stopped"] is True
    assert "stop" in calls
    assert "cache:True" in calls
    assert any("managed multi-GPU vLLM" in n for n in result["release_notes"])


def test_chat_bootstrap_preload_is_opt_in():
    src = Path("forge-ui/src/lib/chatModel.ts").read_text(encoding="utf-8")
    assert "options.preload === true" in src
    assert "options.preload !== false" not in src


def test_interactive_keep_alive_is_short_off_native_linux(monkeypatch):
    from seiso.inference.profiles import profile_sidecar_keep_alive_override

    monkeypatch.setenv("SEISO_INFERENCE_PROFILE", "interactive")
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: False)
    assert profile_sidecar_keep_alive_override() == "2m"


def test_interactive_keep_alive_override_skips_native_linux(monkeypatch):
    from seiso.inference.profiles import profile_sidecar_keep_alive_override

    monkeypatch.setenv("SEISO_INFERENCE_PROFILE", "interactive")
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    assert profile_sidecar_keep_alive_override() is None
