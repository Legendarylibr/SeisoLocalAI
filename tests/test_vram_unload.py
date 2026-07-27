"""Tests for unified inference memory release and VRAM status."""

from __future__ import annotations

import asyncio
import threading

import pytest


@pytest.mark.asyncio
async def test_release_all_inference_memory_unloads_local(monkeypatch, tmp_path):
    from forge.orchestrators import inference as inference_orchestrator

    orchestrator = inference_orchestrator.InferenceOrchestrator(tmp_path)
    calls: list[str] = []

    async def fake_cancel_and_unload():
        calls.append("local")
        return {"active_model": None, "path": None}

    monkeypatch.setattr(orchestrator._runner, "cancel_and_unload", fake_cancel_and_unload)
    monkeypatch.setattr(
        "forge.services.memory_release.release_external_inference_memory",
        lambda **kwargs: (
            calls.append("external"),
            {"release_notes": ["Released Ollama resident models (keep_alive=0)"], "managed_vllm_stopped": True},
        )[1],
    )
    monkeypatch.setattr(
        "forge.services.inference_models.invalidate_inference_options_cache",
        lambda: calls.append("invalidate"),
    )
    monkeypatch.setattr(
        "forge.services.hardware.build_vram_status",
        lambda _o: {"local": {"active_model": None}, "headroom_mb": 8192},
    )

    result = await orchestrator.release_all_inference_memory("user-1")

    assert calls == ["local", "external", "invalidate"]
    assert result["headroom_mb"] == 8192
    assert result["managed_vllm_stopped"] is True
    assert "Released Ollama resident models (keep_alive=0)" in result["release_notes"]


@pytest.mark.asyncio
async def test_cancel_and_unload_for_user_delegates_to_release(monkeypatch, tmp_path):
    from forge.orchestrators import inference as inference_orchestrator

    orchestrator = inference_orchestrator.InferenceOrchestrator(tmp_path)
    seen: list[str] = []

    async def fake_release(user_id):
        seen.append(user_id)
        return {"local": {"active_model": None}}

    monkeypatch.setattr(orchestrator, "release_all_inference_memory", fake_release)

    out = await orchestrator.cancel_and_unload_for_user("u1")
    assert seen == ["u1"]
    assert out["local"]["active_model"] is None


@pytest.mark.asyncio
async def test_release_all_inference_memory_refreshes_headroom(monkeypatch, tmp_path):
    from forge.orchestrators import inference as inference_orchestrator

    orchestrator = inference_orchestrator.InferenceOrchestrator(tmp_path)
    refresh_calls: list[bool] = []

    async def noop_unload():
        return {"active_model": None, "unload_complete": True, "inference_idle": True}

    monkeypatch.setattr(orchestrator._runner, "cancel_and_unload", noop_unload)

    def fake_external(**_kwargs):
        refresh_calls.append(True)
        return {"release_notes": [], "managed_vllm_stopped": False}

    monkeypatch.setattr(
        "forge.services.memory_release.release_external_inference_memory",
        fake_external,
    )
    monkeypatch.setattr(
        "forge.services.inference_models.invalidate_inference_options_cache",
        lambda: None,
    )
    monkeypatch.setattr(
        "forge.services.hardware.build_vram_status",
        lambda _o: {"headroom_mb": 16384, "local": {"active_model": None}},
    )

    result = await orchestrator.release_all_inference_memory(None)
    assert refresh_calls == [True]
    assert result["headroom_mb"] == 16384
    assert result["unload_complete"] is True


def test_build_vram_status_shape(monkeypatch, tmp_path):
    from forge.orchestrators.inference import InferenceOrchestrator
    from forge.services.hardware import build_vram_status

    orchestrator = InferenceOrchestrator(tmp_path)
    monkeypatch.setattr(
        "forge.services.hardware.hardware_profile",
        lambda force_refresh=False: {"ram_gb": 16, "gpus": [], "backend": "metal"},
    )
    monkeypatch.setattr(
        "seiso.hardware.tiers.classify_tier",
        lambda _p: (
            __import__("seiso.hardware.tiers", fromlist=["HardwareTier"]).HardwareTier.APPLE_UNIFIED
        ),
    )
    monkeypatch.setattr("seiso.hardware.tiers.vram_headroom_mb", lambda _p: 10240)
    monkeypatch.setattr("seiso.hardware.memory_headroom_label", lambda _p: "RAM")
    monkeypatch.setattr("seiso.memory.platform_profile.memory_profile_label", lambda _p: "low")
    monkeypatch.setattr(
        orchestrator._runner._pool,
        "status",
        lambda: {"active_model": "model-a", "path": "/tmp/a.gguf"},
    )
    monkeypatch.setattr(
        "forge.services.hardware.recommended_catalog_repo",
        lambda _p, task="chat": "microsoft/Phi-4-mini-instruct",
    )
    monkeypatch.setattr(
        "seiso.hardware.vram_processes.vram_contention_summary",
        lambda: {"external_vram_mb": 0, "contended": False, "processes": []},
    )

    status = build_vram_status(orchestrator)

    assert status["local"]["active_model"] == "model-a"
    assert status["headroom_mb"] == 10240
    assert status["memory_label"] == "RAM"
    assert status["apple_unified"] is True
    assert status["recommended_max_chat"] == "microsoft/Phi-4-mini-instruct"
    assert status["active_model"] == "model-a"
    assert "vram_contention" in status
    assert status["vram_contention"]["external_vram_mb"] == 0


@pytest.mark.asyncio
async def test_preload_cancellation_waits_for_blocking_worker(monkeypatch, tmp_path):
    from forge.orchestrators.inference import InferenceOrchestrator

    orchestrator = InferenceOrchestrator(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def blocking_warm(_payload):
        started.set()
        release.wait(timeout=2)

    monkeypatch.setattr(orchestrator._runner, "warm_model", blocking_warm)
    task = asyncio.create_task(orchestrator.preload_model({"model_path": "x"}))
    await asyncio.to_thread(started.wait, 1)
    task.cancel()
    await asyncio.sleep(0)

    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert orchestrator._preload_future is None


@pytest.mark.asyncio
async def test_preload_excludes_local_chat(monkeypatch, tmp_path):
    from forge.orchestrators.inference import InferenceOrchestrator

    orchestrator = InferenceOrchestrator(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def blocking_warm(_payload):
        started.set()
        release.wait(timeout=2)

    async def fake_chat(_payload):
        return "ok"

    monkeypatch.setattr(orchestrator._runner, "warm_model", blocking_warm)
    monkeypatch.setattr(orchestrator._runner, "chat", fake_chat)
    preload = asyncio.create_task(
        orchestrator.preload_model({"model_path": "x"})
    )
    await asyncio.to_thread(started.wait, 1)
    chat = asyncio.create_task(orchestrator._local_chat({"model_path": "x"}))
    await asyncio.sleep(0)

    assert not chat.done()
    release.set()
    await preload
    assert await chat == "ok"
