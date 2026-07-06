"""Base orchestrator — subprocess workers with SSE log streaming."""

from __future__ import annotations

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from seiso.compat import StrEnum

MAX_LOG_LINES = 2000
MAX_METRIC_POINTS = 5000
MAX_JOBS = 500


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobRecord:
    id: str
    kind: str
    user_id: str | None = None
    status: JobStatus = JobStatus.PENDING
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class Orchestrator(ABC):
    """Spawns isolated worker processes and streams logs via SSE."""

    kind: str = "base"

    def __init__(self, sandbox_root: Path) -> None:
        self.sandbox_root = sandbox_root
        self._jobs: dict[str, JobRecord] = {}
        self._log_buffers: dict[str, deque[str]] = defaultdict(
            lambda: deque(maxlen=MAX_LOG_LINES)
        )
        self._metric_buffers: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=MAX_METRIC_POINTS)
        )
        self._subscribers: dict[str, set[asyncio.Queue[str | None]]] = defaultdict(set)
        self._metric_subscribers: dict[
            str, set[asyncio.Queue[dict[str, Any] | None]]
        ] = defaultdict(set)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._subprocesses: dict[str, asyncio.subprocess.Process] = {}

    def create_job(self, job_id: str | None = None, user_id: str | None = None) -> str:
        if len(self._jobs) >= MAX_JOBS:
            self._evict_oldest_job()
        jid = job_id or str(uuid.uuid4())
        self._jobs[jid] = JobRecord(id=jid, kind=self.kind, user_id=user_id)
        return jid

    def get_job(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def register_subprocess(
        self, job_id: str, proc: asyncio.subprocess.Process
    ) -> None:
        self._subprocesses[job_id] = proc

    def _evict_oldest_job(self) -> None:
        finished = (
            (jid, j)
            for jid, j in self._jobs.items()
            if j.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
        )
        try:
            jid, _ = min(finished, key=lambda x: x[1].created_at)
        except ValueError:
            return
        self._jobs.pop(jid, None)
        self._log_buffers.pop(jid, None)
        self._metric_buffers.pop(jid, None)
        self._tasks.pop(jid, None)
        self._subprocesses.pop(jid, None)

    def _emit_log(self, job_id: str, line: str) -> None:
        buf = self._log_buffers[job_id]
        buf.append(line)
        for q in tuple(self._subscribers.get(job_id, ())):
            q.put_nowait(line)

    def _emit_metric(self, job_id: str, metric: dict[str, Any]) -> None:
        buf = self._metric_buffers[job_id]
        buf.append(metric)
        for q in tuple(self._metric_subscribers.get(job_id, ())):
            q.put_nowait(metric)

    def get_metrics(self, job_id: str) -> list[dict[str, Any]]:
        return list(self._metric_buffers.get(job_id, []))

    def _finish_metrics(self, job_id: str) -> None:
        for q in tuple(self._metric_subscribers.get(job_id, ())):
            q.put_nowait(None)

    async def stream_metrics(self, job_id: str) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        subscribers = self._metric_subscribers[job_id]
        subscribers.add(queue)
        try:
            for point in self._metric_buffers.get(job_id, []):
                yield point
            job = self.get_job(job_id)
            if job and job.status in (
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ):
                return
            while True:
                msg = await queue.get()
                if msg is None:
                    break
                yield msg
        finally:
            subscribers.discard(queue)

    def _finish_logs(self, job_id: str) -> None:
        for q in tuple(self._subscribers.get(job_id, ())):
            q.put_nowait(None)
        self._finish_metrics(job_id)

    async def stream_logs(self, job_id: str) -> AsyncIterator[str]:
        """SSE-compatible log stream for a job."""
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        subscribers = self._subscribers[job_id]
        tail = len(self._log_buffers.get(job_id, []))
        subscribers.add(queue)
        try:
            buf = self._log_buffers.get(job_id, deque())
            while tail < len(buf):
                yield buf[tail]
                tail += 1
            job = self.get_job(job_id)
            if job and job.status in (
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ):
                return
            while True:
                buf = self._log_buffers.get(job_id, deque())
                while tail < len(buf):
                    yield buf[tail]
                    tail += 1
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    job = self.get_job(job_id)
                    if job and job.status in (
                        JobStatus.COMPLETED,
                        JobStatus.FAILED,
                        JobStatus.CANCELLED,
                    ):
                        break
                    continue
                if msg is None:
                    break
        finally:
            subscribers.discard(queue)

    async def start(self, job_id: str, payload: dict[str, Any]) -> None:
        if job_id not in self._jobs:
            raise KeyError(f"Unknown job: {job_id}")
        rec = self._jobs[job_id]
        rec.status = JobStatus.RUNNING

        async def _wrapper() -> None:
            try:
                await self._run(job_id, payload)
            except Exception as exc:
                import logging

                logging.getLogger(__name__).exception("Job %s failed: %s", job_id, exc)

        self._tasks[job_id] = asyncio.create_task(_wrapper())

    async def wait_for(self, job_id: str) -> JobRecord | None:
        """Block until the job task finishes (or was never started)."""
        task = self._tasks.get(job_id)
        if task:
            await task
        return self.get_job(job_id)

    async def _run(self, job_id: str, payload: dict[str, Any]) -> None:
        rec = self._jobs[job_id]
        try:
            result = await self.execute(job_id, payload)
            rec.result = result
            rec.status = JobStatus.COMPLETED
        except asyncio.CancelledError:
            rec.status = JobStatus.CANCELLED
            self._emit_log(job_id, "Job cancelled")
            # Do not re-raise: wait_for must observe CANCELLED and persist it to DB.
        except SystemExit as exc:
            rec.status = JobStatus.FAILED
            rec.error = str(exc) or "Job exited unexpectedly"
            self._emit_log(job_id, f"ERROR: {rec.error}")
        except Exception as exc:
            rec.status = JobStatus.FAILED
            rec.error = str(exc)
            self._emit_log(job_id, f"ERROR: {exc}")
        finally:
            self._subprocesses.pop(job_id, None)
            self._finish_logs(job_id)

    async def cancel(self, job_id: str) -> bool:
        proc = self._subprocesses.get(job_id)
        if proc and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            return True
        return proc is not None

    @abstractmethod
    async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]: ...

    def snapshot(self) -> str:
        return json.dumps(
            {
                jid: {"status": j.status, "kind": j.kind, "error": j.error}
                for jid, j in self._jobs.items()
            }
        )
