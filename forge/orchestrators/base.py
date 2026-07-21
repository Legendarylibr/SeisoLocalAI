"""Base orchestrator — subprocess workers with SSE log streaming."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

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


class ResourceConflictError(RuntimeError):
    """Raised when an orchestrator job would overlap a reserved local resource."""


class JobEventSink(Protocol):
    """Optional durable event sink used by Forge routes for restart-safe streams."""

    def emit(
        self,
        *,
        job_id: str,
        user_id: str,
        kind: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None: ...


_RESOURCE_LOCK = asyncio.Lock()
_ACTIVE_RESOURCES: dict[str, tuple[str, str]] = {}


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
    resource_key: str | None = None

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
        self._subprocess_groups: set[str] = set()
        self._event_sink: JobEventSink | None = None

    def set_event_sink(self, sink: JobEventSink | None) -> None:
        """Attach a durable sink for logs, metrics, status, and result events."""
        self._event_sink = sink

    def create_job(self, job_id: str | None = None, user_id: str | None = None) -> str:
        if len(self._jobs) >= MAX_JOBS:
            self._evict_oldest_job()
        jid = job_id or str(uuid.uuid4())
        self._jobs[jid] = JobRecord(id=jid, kind=self.kind, user_id=user_id)
        self._emit_event(jid, "status", {"status": JobStatus.PENDING.value})
        return jid

    def get_job(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def register_subprocess(
        self,
        job_id: str,
        proc: asyncio.subprocess.Process,
        *,
        process_group: bool = False,
    ) -> None:
        self._subprocesses[job_id] = proc
        if process_group:
            self._subprocess_groups.add(job_id)

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
        self._subprocess_groups.discard(jid)

    def _emit_log(self, job_id: str, line: str) -> None:
        buf = self._log_buffers[job_id]
        buf.append(line)
        self._emit_event(job_id, "log", {"line": line})
        for q in tuple(self._subscribers.get(job_id, ())):
            q.put_nowait(line)

    def _emit_metric(self, job_id: str, metric: dict[str, Any]) -> None:
        buf = self._metric_buffers[job_id]
        buf.append(metric)
        self._emit_event(job_id, "metric", metric)
        for q in tuple(self._metric_subscribers.get(job_id, ())):
            q.put_nowait(metric)

    def _emit_event(
        self, job_id: str, event_type: str, payload: dict[str, Any]
    ) -> None:
        sink = self._event_sink
        rec = self._jobs.get(job_id)
        if sink is None or rec is None or not rec.user_id:
            return
        sink.emit(
            job_id=job_id,
            user_id=rec.user_id,
            kind=self.kind,
            event_type=event_type,
            payload=payload,
        )

    def get_metrics(self, job_id: str) -> list[dict[str, Any]]:
        return list(self._metric_buffers.get(job_id, []))

    def _finish_metrics(self, job_id: str) -> None:
        for q in tuple(self._metric_subscribers.get(job_id, ())):
            q.put_nowait(None)

    async def stream_metrics(
        self, job_id: str, *, replay_buffer: bool = True
    ) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        subscribers = self._metric_subscribers[job_id]
        subscribers.add(queue)
        try:
            if replay_buffer:
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

    async def stream_logs(
        self, job_id: str, *, replay_buffer: bool = True
    ) -> AsyncIterator[str]:
        """SSE-compatible log stream for a job.

        Live lines are yielded from the subscriber queue (not deque indices) so a
        maxlen buffer wrap cannot starve subscribers after ``MAX_LOG_LINES``.
        """
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        subscribers = self._subscribers[job_id]
        subscribers.add(queue)
        try:
            if replay_buffer:
                for line in list(self._log_buffers.get(job_id, ())):
                    yield line
            job = self.get_job(job_id)
            if job and job.status in (
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ):
                return
            while True:
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
                yield msg
        finally:
            subscribers.discard(queue)

    async def start(self, job_id: str, payload: dict[str, Any]) -> None:
        if job_id not in self._jobs:
            raise KeyError(f"Unknown job: {job_id}")
        rec = self._jobs[job_id]
        if rec.status != JobStatus.PENDING:
            raise RuntimeError(f"Job {job_id} is already {rec.status}")
        await self._reserve_resource(job_id, rec)
        rec.status = JobStatus.RUNNING
        self._emit_event(job_id, "status", {"status": JobStatus.RUNNING.value})

        async def _wrapper() -> None:
            try:
                await self._run(job_id, payload)
            except Exception as exc:
                import logging

                logging.getLogger(__name__).exception("Job %s failed: %s", job_id, exc)
            finally:
                await self._release_resource(job_id)

        self._tasks[job_id] = asyncio.create_task(_wrapper())

    async def _reserve_resource(self, job_id: str, rec: JobRecord) -> None:
        if self.resource_key is None:
            return
        async with _RESOURCE_LOCK:
            active = _ACTIVE_RESOURCES.get(self.resource_key)
            if active and active != (self.kind, job_id):
                active_kind, active_job_id = active
                rec.status = JobStatus.FAILED
                rec.error = (
                    f"Cannot start {self.kind} while {active_kind} job "
                    f"{active_job_id} is running"
                )
                self._emit_log(rec.id, f"ERROR: {rec.error}")
                self._emit_event(
                    rec.id,
                    "status",
                    {"status": JobStatus.FAILED.value, "error": rec.error},
                )
                self._finish_logs(rec.id)
                raise ResourceConflictError(rec.error)
            _ACTIVE_RESOURCES[self.resource_key] = (self.kind, job_id)

    async def _release_resource(self, job_id: str) -> None:
        if self.resource_key is None:
            return
        async with _RESOURCE_LOCK:
            active = _ACTIVE_RESOURCES.get(self.resource_key)
            if active == (self.kind, job_id):
                _ACTIVE_RESOURCES.pop(self.resource_key, None)

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
            self._emit_event(
                job_id, "status", {"status": JobStatus.COMPLETED.value}
            )
            self._emit_event(job_id, "result", result)
        except asyncio.CancelledError:
            rec.status = JobStatus.CANCELLED
            self._emit_log(job_id, "Job cancelled")
            self._emit_event(
                job_id, "status", {"status": JobStatus.CANCELLED.value}
            )
            # Do not re-raise: wait_for must observe CANCELLED and persist it to DB.
        except SystemExit as exc:
            rec.status = JobStatus.FAILED
            rec.error = str(exc) or "Job exited unexpectedly"
            self._emit_log(job_id, f"ERROR: {rec.error}")
            self._emit_event(
                job_id,
                "status",
                {"status": JobStatus.FAILED.value, "error": rec.error},
            )
        except Exception as exc:
            rec.status = JobStatus.FAILED
            rec.error = str(exc) or type(exc).__name__
            self._emit_log(job_id, f"ERROR: {rec.error}")
            self._emit_event(
                job_id,
                "status",
                {"status": JobStatus.FAILED.value, "error": rec.error},
            )
        finally:
            self._subprocesses.pop(job_id, None)
            self._subprocess_groups.discard(job_id)
            self._finish_logs(job_id)

    def _terminate_subprocess(self, job_id: str, proc: asyncio.subprocess.Process) -> None:
        if job_id in self._subprocess_groups and os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                return
            except ProcessLookupError:
                return
            except OSError:
                pass
        proc.terminate()

    def _kill_subprocess(self, job_id: str, proc: asyncio.subprocess.Process) -> None:
        if job_id in self._subprocess_groups and os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
                return
            except ProcessLookupError:
                return
            except OSError:
                pass
        proc.kill()

    async def cancel(self, job_id: str) -> bool:
        proc = self._subprocesses.get(job_id)
        if proc and proc.returncode is None:
            self._terminate_subprocess(job_id, proc)
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._kill_subprocess(job_id, proc)
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(proc.wait(), timeout=5)
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
