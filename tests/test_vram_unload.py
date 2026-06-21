"""Tests for unified inference memory release and VRAM status."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_release_all_inference_memory_unloads_local_and_ollama(monkeypatch, tmp_path):
    from forge.orchestrators import inference as inference_orchestrator

    orchestrator = inference_orchestrator.InferenceOrchestrator(tmp_path)
    orchestrator._active_ollama_model = "llama3.2"
    orchestrator._active_ollama_base_url = "http://127.0.0.1:11434"
    calls: list[str] = []

    async def fake_cancel_and_unload():
        calls.append("local")
        return {"active_model": None, "path": None}

    async def fake_release_ollama():
        calls.append("ollama")
        orchestrator._active_ollama_model = None

    monkeypatch.setattr(orchestrator._runner, "cancel_and_unload", fake_cancel_and_unload)
    monkeypatch.setattr(orchestrator, "_release_ollama_model", fake_release_ollama)
    monkeypatch.setattr(
        "seiso.memory.protection.release_cached_memory",
        lambda sync=False: calls.append(f"cache:{sync}"),
    )
    monkeypatch.setattr(
        "seiso.hardware.profile.hardware_profile", lambda force_refresh=False: {"ram_gb": 16}
    )
    monkeypatch.setattr(
        "forge.services.inference_models.invalidate_inference_options_cache",
        lambda: calls.append("invalidate"),
    )
    monkeypatch.setattr(
        "forge.services.hardware.build_vram_status",
        lambda _o: {"local": {"active_model": None}, "ollama_model": None, "headroom_mb": 8192},
    )

    result = await orchestrator.release_all_inference_memory("user-1")

    assert calls == ["local", "ollama", "cache:True", "invalidate"]
    assert result["headroom_mb"] == 8192
    assert orchestrator.active_ollama_model is None


@pytest.mark.asyncio
async def test_cancel_and_unload_for_user_delegates_to_release(monkeypatch, tmp_path):
    from forge.orchestrators import inference as inference_orchestrator

    orchestrator = inference_orchestrator.InferenceOrchestrator(tmp_path)
    seen: list[str] = []

    async def fake_release(user_id):
        seen.append(user_id)
        return {"local": {"active_model": None}, "ollama_model": None}

    monkeypatch.setattr(orchestrator, "release_all_inference_memory", fake_release)

    out = await orchestrator.cancel_and_unload_for_user("u1")
    assert seen == ["u1"]
    assert out["ollama_model"] is None


@pytest.mark.asyncio
async def test_release_all_inference_memory_refreshes_headroom(monkeypatch, tmp_path):
    from forge.orchestrators import inference as inference_orchestrator

    orchestrator = inference_orchestrator.InferenceOrchestrator(tmp_path)
    refresh_calls: list[bool] = []

    async def noop_unload():
        return {"active_model": None}

    monkeypatch.setattr(orchestrator._runner, "cancel_and_unload", noop_unload)
    monkeypatch.setattr(orchestrator, "_release_ollama_model", noop_unload)
    monkeypatch.setattr("seiso.memory.protection.release_cached_memory", lambda sync=False: None)

    def fake_hw(force_refresh=False):
        refresh_calls.append(force_refresh)
        return {"ram_gb": 24, "gpus": []}

    monkeypatch.setattr("seiso.hardware.profile.hardware_profile", fake_hw)
    monkeypatch.setattr(
        "forge.services.inference_models.invalidate_inference_options_cache", lambda: None
    )
    monkeypatch.setattr(
        "forge.services.hardware.build_vram_status",
        lambda _o: {"headroom_mb": 16384, "local": {"active_model": None}, "ollama_model": None},
    )

    result = await orchestrator.release_all_inference_memory(None)
    assert refresh_calls == [True]
    assert result["headroom_mb"] == 16384


def test_build_vram_status_shape(monkeypatch, tmp_path):
    from forge.orchestrators.inference import InferenceOrchestrator
    from forge.services.hardware import build_vram_status

    orchestrator = InferenceOrchestrator(tmp_path)
    orchestrator._active_ollama_model = "mistral"
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
        "forge.services.hardware.largest_fitting_catalog_repo",
        lambda _p, task="chat": "microsoft/Phi-4-mini-instruct",
    )

    status = build_vram_status(orchestrator)

    assert status["ollama_model"] == "mistral"
    assert status["local"]["active_model"] == "model-a"
    assert status["headroom_mb"] == 10240
    assert status["memory_label"] == "RAM"
    assert status["apple_unified"] is True
    assert status["recommended_max_chat"] == "microsoft/Phi-4-mini-instruct"
    assert status["active_model"] == "model-a"
