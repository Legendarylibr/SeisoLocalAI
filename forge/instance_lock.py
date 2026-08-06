"""Single Forge instance guards — port slot lock and data-dir flock."""

from __future__ import annotations

import atexit
import errno
import getpass
import json
import os
import socket
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from seiso.io.jsonl import read_json_file

_WINDOWS_ADDR_IN_USE = frozenset({10048, 10049})


class ForgeAlreadyRunningError(RuntimeError):
    """Raised when another Forge instance holds the port or data directory."""

    def __init__(self, message: str, *, url: str | None = None, pid: int | None = None) -> None:
        super().__init__(message)
        self.url = url
        self.pid = pid


def multi_forge_allowed() -> bool:
    return os.environ.get("SEISO_ALLOW_MULTI_FORGE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def data_dir_lock_path(data_dir: Path) -> Path:
    return data_dir / ".forge.lock"


def port_lock_path(host: str, port: int) -> Path:
    safe_host = host.replace(":", "_").replace("/", "_") or "localhost"
    user = getpass.getuser()
    base = Path(tempfile.gettempdir()) / "seiso-forge-locks" / user
    return base / f"{safe_host}-{port}.lock"


def lock_held_by_current_process(path: Path) -> bool:
    meta = _read_lock_meta(path)
    return int(meta.get("pid") or 0) == os.getpid()


def _read_lock_meta(path: Path) -> dict[str, object]:
    data = read_json_file(path, default={})
    return data if isinstance(data, dict) else {}


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def _health_is_seiso(url: str, *, timeout: float = 0.5) -> bool:
    health = f"{url.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(health, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _is_addr_in_use(exc: OSError) -> bool:
    if exc.errno in {errno.EADDRINUSE, errno.EADDRNOTAVAIL}:
        return True
    return getattr(exc, "winerror", None) in _WINDOWS_ADDR_IN_USE


def _probe_bind(host: str, port: int) -> None:
    bind_host = host
    if host in {
        "0.0.0.0",
        "::",
    }:  # nosec B104 — detect all-interfaces bind requests, not binding to them
        bind_host = "127.0.0.1"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_REUSEADDR"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((bind_host, port))
    except OSError as exc:
        if _is_addr_in_use(exc):
            raise exc
        raise
    finally:
        sock.close()


def assert_port_available(host: str, port: int) -> None:
    """Fail fast when host:port is already bound or serving Forge."""
    if multi_forge_allowed():
        return

    url = f"http://{host}:{port}"
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.5)
    try:
        probe.connect((host, port))
    except (ConnectionRefusedError, TimeoutError, OSError):
        pass
    else:
        if _health_is_seiso(url):
            raise ForgeAlreadyRunningError(
                f"Forge is already running at {url}. "
                "Stop it first or set SEISO_PORT to a different value.",
                url=url,
            )
        raise ForgeAlreadyRunningError(
            f"Port {port} on {host} is already in use by another process.",
            url=url,
        )
    finally:
        probe.close()

    try:
        _probe_bind(host, port)
    except OSError as exc:
        if not _is_addr_in_use(exc):
            raise
        message = f"Forge is already running — cannot bind {host}:{port}."
        if _health_is_seiso(url):
            message = (
                f"Forge is already running at {url}. "
                "Stop it first or set SEISO_PORT to a different value."
            )
        raise ForgeAlreadyRunningError(message, url=url) from exc


def _is_lock_contention(exc: BaseException) -> bool:
    if isinstance(exc, BlockingIOError):
        return True
    if isinstance(exc, OSError) and sys.platform == "win32":
        return getattr(exc, "winerror", None) in {33, 32}
    return False


def _flock_exclusive_nb(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt  # pylint: disable=import-error

        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _flock_unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt  # pylint: disable=import-error

        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


class _ExclusiveFileLock:
    def __init__(self) -> None:
        self._fd: int | None = None
        self._released = False

    def _try_acquire(
        self,
        path: Path,
        *,
        meta: dict[str, object],
        busy_factory: Callable[[dict[str, object]], ForgeAlreadyRunningError],
        retry_stale: bool,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            _flock_exclusive_nb(fd)
        except (BlockingIOError, OSError) as exc:
            if not _is_lock_contention(exc):
                os.close(fd)
                raise
            existing = _read_lock_meta(path)
            pid = int(existing.get("pid") or 0)
            if retry_stale and pid and not _pid_alive(pid):
                os.close(fd)
                path.unlink(missing_ok=True)
                self._try_acquire(path, meta=meta, busy_factory=busy_factory, retry_stale=False)
                return
            os.close(fd)
            raise busy_factory(existing) from exc

        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, json.dumps(meta).encode("utf-8"))
        self._fd = fd
        self._released = False
        atexit.register(self.release)

    def release(self) -> None:
        if self._released or self._fd is None:
            return
        fd = self._fd
        self._fd = None
        self._released = True
        try:
            _flock_unlock(fd)
        finally:
            os.close(fd)


class ForgePortLock(_ExclusiveFileLock):
    """Machine-wide exclusive lock for a host:port Forge slot."""

    def acquire(self, host: str, port: int) -> None:
        if multi_forge_allowed():
            return
        url = f"http://{host}:{port}"
        path = port_lock_path(host, port)
        meta = {
            "pid": os.getpid(),
            "host": host,
            "port": port,
            "url": url,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        def _busy(existing: dict[str, object]) -> ForgeAlreadyRunningError:
            other_url = str(existing.get("url") or url)
            other_pid = int(existing.get("pid") or 0) or None
            return ForgeAlreadyRunningError(
                f"Forge is already running at {other_url} (pid {other_pid or 'unknown'}). "
                f"Stop it first — only one backend can use {host}:{port}.",
                url=other_url,
                pid=other_pid,
            )

        self._try_acquire(path, meta=meta, busy_factory=_busy, retry_stale=True)


class ForgeDataDirLock(_ExclusiveFileLock):
    """Exclusive flock on {data_dir}/.forge.lock — one Forge per data directory."""

    def acquire(self, data_dir: Path, *, host: str, port: int) -> None:
        if multi_forge_allowed():
            return
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir_lock_path(data_dir)
        url = f"http://{host}:{port}"
        meta = {
            "pid": os.getpid(),
            "host": host,
            "port": port,
            "url": url,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        def _busy(existing: dict[str, object]) -> ForgeAlreadyRunningError:
            other_url = str(existing.get("url") or url)
            other_pid = int(existing.get("pid") or 0) or None
            return ForgeAlreadyRunningError(
                f"Another Forge instance holds this data directory "
                f"(pid {other_pid or 'unknown'} at {other_url}). "
                "Stop it first or use a different SEISO_DATA_DIR.",
                url=other_url,
                pid=other_pid,
            )

        self._try_acquire(path, meta=meta, busy_factory=_busy, retry_stale=True)


@dataclass
class ForgeInstanceLocks:
    port_lock: ForgePortLock | None = field(default=None)
    data_lock: ForgeDataDirLock | None = field(default=None)

    def release(self) -> None:
        if self.data_lock is not None:
            self.data_lock.release()
            self.data_lock = None
        if self.port_lock is not None:
            self.port_lock.release()
            self.port_lock = None


def acquire_forge_instance_locks(*, host: str, port: int, data_dir: Path) -> ForgeInstanceLocks:
    """Acquire port + data-dir locks for `seiso forge` — held until release()."""
    if multi_forge_allowed():
        return ForgeInstanceLocks()

    assert_port_available(host, port)
    locks = ForgeInstanceLocks()
    locks.port_lock = ForgePortLock()
    locks.port_lock.acquire(host, port)
    locks.data_lock = ForgeDataDirLock()
    try:
        locks.data_lock.acquire(data_dir, host=host, port=port)
    except Exception:
        locks.release()
        raise
    return locks
