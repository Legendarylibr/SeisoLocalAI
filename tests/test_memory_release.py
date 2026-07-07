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

    memory_release._ACTIVE_GPU_TASKS.clear()
    monkeypatch.setattr(memory_release, "release_inference_memory", lambda **kwargs: {})
    lock_events: list[str] = []
    monkeypatch.setattr(
        "seiso.memory.gpu_resource_lock.acquire_gpu_resource_lock",
        lambda: lock_events.append("acquire"),
    )
    monkeypatch.setattr(
        "seiso.memory.gpu_resource_lock.release_gpu_resource_lock",
        lambda: lock_events.append("release"),
    )
    monkeypatch.setattr(
        memory_release,
        "running_gpu_task_kinds",
        lambda exclude_job_id=None: (
            [] if exclude_job_id in {"job-1", "job-2"} else ["training"]
        ),
    )

    result = memory_release.prepare_for_gpu_task(task="export", job_id="job-2")
    memory_release.release_after_task(reason="export complete", job_id="job-2")
    assert result["resource_token"] == "job-2"
    assert lock_events == ["acquire", "release"]

    with pytest.raises(RuntimeError, match="Another GPU task"):
        memory_release.prepare_for_gpu_task(task="export", job_id="job-3")
    memory_release._ACTIVE_GPU_TASKS.clear()


def test_assert_gpu_available_for_inference_blocks_training(monkeypatch):
    from forge.services.memory_release import assert_gpu_available_for_inference

    monkeypatch.setattr(
        "forge.services.memory_release.running_gpu_task_kinds",
        lambda exclude_job_id=None: ["training"],
    )

    with pytest.raises(RuntimeError, match="Cannot load chat models"):
        assert_gpu_available_for_inference()


def test_download_resource_blocks_inference_and_gpu_tasks(monkeypatch):
    from forge.services import memory_release

    memory_release._ACTIVE_GPU_TASKS.clear()
    monkeypatch.setattr(
        memory_release,
        "running_gpu_task_kinds",
        memory_release.running_gpu_task_kinds,
    )
    monkeypatch.setattr(memory_release, "release_inference_memory", lambda **kwargs: {})
    lock_events: list[str] = []
    monkeypatch.setattr(
        "seiso.memory.gpu_resource_lock.acquire_gpu_resource_lock",
        lambda: lock_events.append("acquire"),
    )
    monkeypatch.setattr(
        "seiso.memory.gpu_resource_lock.release_gpu_resource_lock",
        lambda: lock_events.append("release"),
    )
    monkeypatch.setattr(
        "seiso.memory.protection.release_cached_memory",
        lambda sync=False: None,
    )
    monkeypatch.setattr(memory_release, "_refresh_hardware_profile", lambda: None)

    result = memory_release.prepare_for_gpu_task(task="download", user_id="u1")

    assert result["resource_token"].startswith("download:u1:")
    assert memory_release.running_gpu_task_kinds() == ["download"]
    with pytest.raises(RuntimeError, match="Cannot load chat models while download"):
        memory_release.assert_gpu_available_for_inference()
    with pytest.raises(RuntimeError, match="Another GPU task is still running"):
        memory_release.prepare_for_gpu_task(task="training", job_id="train-1")

    memory_release.release_after_task(
        reason="download complete",
        resource_token=result["resource_token"],
    )

    assert memory_release.running_gpu_task_kinds() == []
    assert lock_events == ["acquire", "release"]


def test_prepare_for_gpu_task_releases_lock_on_unload_failure(monkeypatch):
    from forge.services import memory_release

    memory_release._ACTIVE_GPU_TASKS.clear()
    lock_events: list[str] = []
    monkeypatch.setattr(memory_release, "running_gpu_task_kinds", lambda **_kwargs: [])
    monkeypatch.setattr(
        "seiso.memory.gpu_resource_lock.acquire_gpu_resource_lock",
        lambda: lock_events.append("acquire"),
    )
    monkeypatch.setattr(
        "seiso.memory.gpu_resource_lock.release_gpu_resource_lock",
        lambda: lock_events.append("release"),
    )
    monkeypatch.setattr(
        memory_release,
        "release_inference_memory",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("unload failed")),
    )

    with pytest.raises(RuntimeError, match="unload failed"):
        memory_release.prepare_for_gpu_task(task="training", job_id="train-1")

    assert lock_events == ["acquire", "release"]
    assert memory_release.running_gpu_task_kinds() == []
