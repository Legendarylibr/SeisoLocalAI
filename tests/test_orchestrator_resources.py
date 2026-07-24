from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from typing import Any

import pytest

from forge.orchestrators import base
from forge.orchestrators.base import JobStatus, Orchestrator, ResourceConflictError


class _GpuOrchestratorA(Orchestrator):
    kind = "gpu-a"
    resource_key = "gpu"

    def __init__(self, sandbox_root: Path, started: asyncio.Event, stop: asyncio.Event):
        super().__init__(sandbox_root)
        self.started = started
        self.stop = stop

    async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.started.set()
        await self.stop.wait()
        return {"job_id": job_id}


class _GpuOrchestratorB(Orchestrator):
    kind = "gpu-b"
    resource_key = "gpu"

    async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"job_id": job_id}


@pytest.mark.asyncio
async def test_gpu_resource_jobs_do_not_overlap(tmp_path: Path):
    started = asyncio.Event()
    stop = asyncio.Event()
    first = _GpuOrchestratorA(tmp_path, started, stop)
    second = _GpuOrchestratorB(tmp_path)

    first_job = first.create_job(user_id="user-a")
    await first.start(first_job, {})
    await asyncio.wait_for(started.wait(), timeout=2)

    second_job = second.create_job(user_id="user-a")
    with pytest.raises(ResourceConflictError):
        await second.start(second_job, {})
    assert second.get_job(second_job).status == JobStatus.FAILED

    stop.set()
    await first.wait_for(first_job)

    retry_job = second.create_job(user_id="user-a")
    await second.start(retry_job, {})
    retry = await second.wait_for(retry_job)
    assert retry is not None
    assert retry.status == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_job_runtime_persists_resource_conflict_failure(tmp_path: Path):
    from forge.services.job_runtime import run_orchestrated_job

    started = asyncio.Event()
    stop = asyncio.Event()
    first = _GpuOrchestratorA(tmp_path, started, stop)
    second = _GpuOrchestratorB(tmp_path)
    failures: list[str] = []

    first_job = first.create_job(user_id="user-a")
    await first.start(first_job, {})
    await asyncio.wait_for(started.wait(), timeout=2)

    second_job = second.create_job(user_id="user-a")
    await run_orchestrated_job(
        orchestrator=second,
        job_id=second_job,
        payload={},
        on_finished=lambda _job: asyncio.sleep(0),
        on_failed=lambda message: asyncio.sleep(0, result=failures.append(message)),
    )

    stop.set()
    await first.wait_for(first_job)
    assert failures and "Cannot start gpu-b while gpu-a job" in failures[0]


@pytest.mark.asyncio
async def test_job_start_is_single_use(tmp_path: Path):
    orchestrator = _GpuOrchestratorB(tmp_path)
    job_id = orchestrator.create_job(user_id="user-a")

    await orchestrator.start(job_id, {})
    await orchestrator.wait_for(job_id)

    with pytest.raises(RuntimeError):
        await orchestrator.start(job_id, {})


@pytest.mark.asyncio
async def test_cancel_pending_job_before_start(tmp_path: Path):
    orchestrator = _GpuOrchestratorB(tmp_path)
    job_id = orchestrator.create_job(user_id="user-a")
    assert orchestrator.get_job(job_id).status == JobStatus.PENDING
    assert await orchestrator.cancel(job_id) is True
    assert orchestrator.get_job(job_id).status == JobStatus.CANCELLED
    with pytest.raises(RuntimeError, match="already cancelled"):
        await orchestrator.start(job_id, {})


@pytest.mark.asyncio
async def test_cancel_during_resource_reserve_does_not_run(tmp_path: Path, monkeypatch):
    """Cancel during start()'s await must stick; job must not become RUNNING."""
    orchestrator = _GpuOrchestratorB(tmp_path)
    job_id = orchestrator.create_job(user_id="user-a")
    started = asyncio.Event()
    release = asyncio.Event()
    orig_reserve = orchestrator._reserve_resource

    async def slow_reserve(job_id_arg, rec):
        started.set()
        await release.wait()
        return await orig_reserve(job_id_arg, rec)

    monkeypatch.setattr(orchestrator, "_reserve_resource", slow_reserve)
    task = asyncio.create_task(orchestrator.start(job_id, {}))
    await started.wait()
    assert await orchestrator.cancel(job_id) is True
    release.set()
    await task
    rec = orchestrator.get_job(job_id)
    assert rec is not None
    assert rec.status == JobStatus.CANCELLED
    assert rec.cancel_requested is True
    assert job_id not in orchestrator._tasks


@pytest.mark.asyncio
async def test_cancel_terminates_registered_process_group(monkeypatch, tmp_path: Path):
    orchestrator = _GpuOrchestratorB(tmp_path)
    job_id = orchestrator.create_job(user_id="user-a")
    calls: list[tuple[int, signal.Signals]] = []

    class FakeProc:
        pid = 4242
        returncode: int | None = None

        def terminate(self) -> None:
            raise AssertionError("process group should be terminated instead")

        def kill(self) -> None:
            raise AssertionError("process group should be killed instead")

        async def wait(self) -> int:
            self.returncode = -int(signal.SIGTERM)
            return self.returncode

    proc = FakeProc()
    monkeypatch.setattr(
        base.os,
        "killpg",
        lambda pid, sig: calls.append((pid, signal.Signals(sig))),
    )

    orchestrator.register_subprocess(job_id, proc, process_group=True)  # type: ignore[arg-type]

    assert await orchestrator.cancel(job_id) is True
    assert calls == [(4242, signal.SIGTERM)]


def test_begin_generation_blocks_while_backend_refs_active(tmp_path: Path):
    from forge.orchestrators.inference import InferenceOrchestrator

    orchestrator = InferenceOrchestrator(tmp_path)
    pool = orchestrator._runner.pool
    pool.begin_inference()
    try:
        with pytest.raises(RuntimeError, match="still stopping"):
            orchestrator.begin_generation_for_user("user-a")
    finally:
        pool.end_inference()


@pytest.mark.asyncio
async def test_stream_local_updates_keeps_outer_generation_hold(tmp_path: Path, monkeypatch):
    """Auto-continue multi-pass must not release the route-level generation hold.

    Releasing between passes re-enters begin_generation and can raise
    'backend is still stopping' when pool refs have not drained yet — aborting
    long replies mid-stream.
    """
    from forge.orchestrators.inference import InferenceOrchestrator
    from seiso.inference.streaming import StreamUpdate

    orchestrator = InferenceOrchestrator(tmp_path)
    orchestrator.begin_generation_for_user("user-a")

    async def fake_stream_updates(_payload):
        yield StreamUpdate(text="chunk", output_tokens=1, metadata={})

    monkeypatch.setattr(orchestrator._runner, "stream_updates", fake_stream_updates)

    tokens = [
        update.text
        async for update in orchestrator.stream_local_updates(
            {"user_id": "user-a", "max_tokens": 8}
        )
    ]
    assert tokens == ["chunk"]
    # Outer reservation must still be held after the first pass.
    assert orchestrator._active_generation_user_id == "user-a"

    # A second continue pass must also leave the hold in place.
    tokens2 = [
        update.text
        async for update in orchestrator.stream_local_updates(
            {"user_id": "user-a", "max_tokens": 8}
        )
    ]
    assert tokens2 == ["chunk"]
    assert orchestrator._active_generation_user_id == "user-a"
    orchestrator.end_generation_for_user("user-a")


@pytest.mark.asyncio
async def test_stream_local_updates_releases_when_it_began(tmp_path: Path, monkeypatch):
    from forge.orchestrators.inference import InferenceOrchestrator
    from seiso.inference.streaming import StreamUpdate

    orchestrator = InferenceOrchestrator(tmp_path)

    async def fake_stream_updates(_payload):
        yield StreamUpdate(text="solo", output_tokens=1, metadata={})

    monkeypatch.setattr(orchestrator._runner, "stream_updates", fake_stream_updates)

    tokens = [
        update.text
        async for update in orchestrator.stream_local_updates(
            {"user_id": "user-b", "max_tokens": 8}
        )
    ]
    assert tokens == ["solo"]
    assert orchestrator._active_generation_user_id is None
