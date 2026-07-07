from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

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
