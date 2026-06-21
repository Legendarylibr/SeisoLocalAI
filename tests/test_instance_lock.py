"""Tests for Forge single-instance guards."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest

from forge.instance_lock import (
    ForgeAlreadyRunningError,
    ForgeDataDirLock,
    ForgePortLock,
    acquire_forge_instance_locks,
    assert_port_available,
    multi_forge_allowed,
)


def test_multi_forge_allowed_env(monkeypatch):
    monkeypatch.delenv("SEISO_ALLOW_MULTI_FORGE", raising=False)
    assert multi_forge_allowed() is False
    monkeypatch.setenv("SEISO_ALLOW_MULTI_FORGE", "1")
    assert multi_forge_allowed() is True


def test_assert_port_available_on_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    assert_port_available("127.0.0.1", port)


def test_assert_port_available_rejects_bound_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.listen(1)
    try:
        with pytest.raises(ForgeAlreadyRunningError, match="already in use"):
            assert_port_available("127.0.0.1", port)
    finally:
        sock.close()


def test_data_dir_lock_blocks_second_acquire(tmp_path: Path):
    lock_a = ForgeDataDirLock()
    lock_b = ForgeDataDirLock()
    lock_a.acquire(tmp_path, host="127.0.0.1", port=8765)
    try:
        with pytest.raises(ForgeAlreadyRunningError, match="data directory"):
            lock_b.acquire(tmp_path, host="127.0.0.1", port=8766)
    finally:
        lock_a.release()


def test_data_dir_lock_stale_pid_recovery(tmp_path: Path, monkeypatch):
    lock_path = tmp_path / ".forge.lock"
    lock_path.write_text(
        json.dumps({"pid": 999999999, "host": "127.0.0.1", "port": 8765, "url": "http://127.0.0.1:8765"}),
        encoding="utf-8",
    )
    lock = ForgeDataDirLock()
    lock.acquire(tmp_path, host="127.0.0.1", port=8765)
    try:
        assert lock_path.exists()
        meta = json.loads(lock_path.read_text(encoding="utf-8"))
        assert meta["pid"] == os.getpid()
    finally:
        lock.release()


def test_assert_port_available_skipped_when_multi_forge(monkeypatch):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.listen(1)
    monkeypatch.setenv("SEISO_ALLOW_MULTI_FORGE", "1")
    try:
        assert_port_available("127.0.0.1", port)
    finally:
        sock.close()


def test_port_lock_blocks_second_instance_on_same_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    lock_a = ForgePortLock()
    lock_b = ForgePortLock()
    lock_a.acquire("127.0.0.1", port)
    try:
        with pytest.raises(ForgeAlreadyRunningError, match="only one backend"):
            lock_b.acquire("127.0.0.1", port)
    finally:
        lock_a.release()
        sock.close()


def test_acquire_forge_instance_locks_holds_both(tmp_path: Path):
    locks = acquire_forge_instance_locks(host="127.0.0.1", port=18765, data_dir=tmp_path)
    try:
        with pytest.raises(ForgeAlreadyRunningError):
            acquire_forge_instance_locks(host="127.0.0.1", port=18765, data_dir=tmp_path / "other")
    finally:
        locks.release()
