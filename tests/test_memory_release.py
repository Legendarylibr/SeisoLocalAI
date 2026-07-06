"""Tests for unified GPU/RAM release before Forge tasks."""

from __future__ import annotations

import pytest


def test_release_inference_memory_unloads_active(monkeypatch):
    from forge.services import memory_release

    calls: list[str] = []

    class FakePool:
        active_key = "model-a"

        def status(self):
            return {"active_model": "model-a", "path": "/tmp/a.gguf"}

        def prepare_for_load(self, target_path=None, backend=None):
            calls.append("unload")
            self.active_key = None
            return True

    fake = FakePool()
    monkeypatch.setattr(
        "seiso.inference.model_pool.get_model_pool",
        lambda: fake,
    )
    monkeypatch.setattr(
        "seiso.memory.protection.release_cached_memory",
        lambda sync=False: calls.append(f"cache:{sync}"),
    )
    monkeypatch.setattr(
        memory_release, "_refresh_hardware_profile", lambda: calls.append("refresh")
    )

    result = memory_release.release_inference_memory(reason="training")

    assert result["unloaded_inference"] is True
    assert "unload" in calls
    assert "cache:True" in calls


def test_prepare_for_gpu_task_blocks_other_running_jobs(monkeypatch):
    from forge.services import memory_release

    monkeypatch.setattr(memory_release, "release_inference_memory", lambda **kwargs: {})
    monkeypatch.setattr(
        memory_release,
        "running_gpu_task_kinds",
        lambda exclude_job_id=None: (
            [] if exclude_job_id in {"job-1", "job-2"} else ["training"]
        ),
    )

    memory_release.prepare_for_gpu_task(task="export", job_id="job-2")

    with pytest.raises(RuntimeError, match="Another GPU task"):
        memory_release.prepare_for_gpu_task(task="export", job_id="job-3")


def test_assert_gpu_available_for_inference_blocks_training(monkeypatch):
    from forge.services.memory_release import assert_gpu_available_for_inference

    monkeypatch.setattr(
        "forge.services.memory_release.running_gpu_task_kinds",
        lambda exclude_job_id=None: ["training"],
    )

    with pytest.raises(RuntimeError, match="Cannot load chat models"):
        assert_gpu_available_for_inference()
