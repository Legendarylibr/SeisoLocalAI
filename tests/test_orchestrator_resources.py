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
async def test_job_start_is_single_use(tmp_path: Path):
    orchestrator = _GpuOrchestratorB(tmp_path)
    job_id = orchestrator.create_job(user_id="user-a")

    await orchestrator.start(job_id, {})
    await orchestrator.wait_for(job_id)

    with pytest.raises(RuntimeError):
        await orchestrator.start(job_id, {})


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
