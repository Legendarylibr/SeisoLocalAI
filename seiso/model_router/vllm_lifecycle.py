"""vLLM sleep/wake lifecycle and idle management."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

from seiso.model_router.catalog import SpecialistCatalog, SpecialistRoute
from seiso.model_router.config import RouterSettings

logger = logging.getLogger(__name__)


class BackendState(str, Enum):
    UNKNOWN = "unknown"
    AWAKE = "awake"
    SLEEPING = "sleeping"
    UNREACHABLE = "unreachable"


@dataclass
class BackendRecord:
    route_id: str
    vllm_url: str
    state: BackendState = BackendState.UNKNOWN
    last_used: float = 0.0
    last_sleep_attempt: float = 0.0
    wake_latency_ms: float = 0.0
    error: str = ""


@dataclass
class VLLMLifecycleManager:
    settings: RouterSettings
    catalog: SpecialistCatalog
    _records: dict[str, BackendRecord] = field(default_factory=dict, init=False)
    _client: httpx.AsyncClient | None = field(default=None, init=False)
    _poll_task: asyncio.Task[None] | None = field(default=None, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        for route in self.catalog:
            self._records[route.route_id] = BackendRecord(
                route_id=route.route_id,
                vllm_url=route.vllm_url,
            )

    async def start(self) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.settings.request_timeout_sec))
        self._poll_task = asyncio.create_task(self._idle_poll_loop())
        await self._refresh_all_states()

    async def stop(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()

    async def _idle_poll_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.settings.lifecycle_poll_sec)
                await self._sleep_idle_backends()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("idle poll failed")

    async def _refresh_all_states(self) -> None:
        for route in self.catalog:
            await self._probe_state(route)

    async def _probe_state(self, route: SpecialistRoute) -> BackendState:
        client = self._client
        if client is None:
            return BackendState.UNKNOWN
        record = self._records[route.route_id]
        try:
            resp = await client.get(f"{route.vllm_url}/is_sleeping", timeout=5.0)
            if resp.status_code == 404:
                # Sleep API not enabled — assume awake if health ok
                health = await client.get(f"{route.vllm_url}/health", timeout=5.0)
                record.state = (
                    BackendState.AWAKE if health.status_code == 200 else BackendState.UNREACHABLE
                )
            elif resp.status_code == 200:
                data = resp.json()
                sleeping = bool(data.get("is_sleeping", data.get("sleeping", False)))
                record.state = BackendState.SLEEPING if sleeping else BackendState.AWAKE
            else:
                record.state = BackendState.UNREACHABLE
            record.error = ""
        except Exception as exc:
            record.state = BackendState.UNREACHABLE
            record.error = str(exc)
        return record.state

    async def wake_for_route(self, route: SpecialistRoute) -> BackendRecord:
        async with self._lock:
            record = self._records[route.route_id]
            record.last_used = time.monotonic()
            if route.vram_hot:
                record.state = BackendState.AWAKE
                return record

            state = await self._probe_state(route)
            if state == BackendState.SLEEPING:
                await self._wake(route, record)
            elif state == BackendState.UNREACHABLE:
                # Retry wake in case process is starting
                await self._wake(route, record)
            return record

    async def _wake(self, route: SpecialistRoute, record: BackendRecord) -> None:
        client = self._client
        if client is None:
            return
        start = time.monotonic()
        try:
            resp = await client.post(f"{route.vllm_url}/wake_up", timeout=self.settings.wake_timeout_sec)
            if resp.status_code >= 400:
                # Query param fallback
                resp = await client.post(
                    f"{route.vllm_url}/wake_up?tags=weights",
                    timeout=self.settings.wake_timeout_sec,
                )
            record.wake_latency_ms = (time.monotonic() - start) * 1000.0
            record.state = BackendState.AWAKE if resp.status_code < 400 else BackendState.UNREACHABLE
            record.error = "" if resp.status_code < 400 else resp.text[:200]
        except Exception as exc:
            record.state = BackendState.UNREACHABLE
            record.error = str(exc)

    async def sleep_route(self, route: SpecialistRoute) -> None:
        if route.vram_hot:
            return
        client = self._client
        if client is None:
            return
        record = self._records[route.route_id]
        level = max(1, min(2, route.sleep_level))
        try:
            resp = await client.post(
                f"{route.vllm_url}/sleep",
                json={"level": str(level)},
                timeout=60.0,
            )
            if resp.status_code >= 400:
                resp = await client.post(
                    f"{route.vllm_url}/sleep?level={level}",
                    timeout=60.0,
                )
            record.state = BackendState.SLEEPING if resp.status_code < 400 else record.state
            record.last_sleep_attempt = time.monotonic()
        except Exception as exc:
            record.error = str(exc)
            logger.warning("sleep failed for %s: %s", route.route_id, exc)

    async def _sleep_idle_backends(self) -> None:
        now = time.monotonic()
        hot_ids = self._vram_hot_route_ids()
        for route in self.catalog:
            if route.route_id in hot_ids:
                continue
            idle_sec = route.effective_idle_sec(self.settings.default_idle_sleep_sec)
            if idle_sec <= 0:
                continue
            record = self._records[route.route_id]
            if record.state != BackendState.AWAKE:
                continue
            if record.last_used <= 0:
                continue
            if now - record.last_used < idle_sec:
                continue
            await self.sleep_route(route)

    def _vram_hot_route_ids(self) -> set[str]:
        hot = [r for r in self.catalog if r.vram_hot]
        hot.sort(key=lambda r: self._records[r.route_id].last_used, reverse=True)
        limit = max(0, self.settings.max_vram_hot)
        return {r.route_id for r in hot[:limit]}

    def touch(self, route_id: str) -> None:
        record = self._records.get(route_id)
        if record:
            record.last_used = time.monotonic()

    def status(self) -> dict[str, Any]:
        return {
            "backends": {
                rid: {
                    "state": rec.state.value,
                    "vllm_url": rec.vllm_url,
                    "last_used": rec.last_used,
                    "wake_latency_ms": rec.wake_latency_ms,
                    "error": rec.error,
                    "vram_hot": self.catalog.by_id(rid).vram_hot,
                }
                for rid, rec in self._records.items()
            },
            "vram_hot_active": list(self._vram_hot_route_ids()),
        }
