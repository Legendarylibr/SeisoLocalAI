"""JWT JTI revocation — expiry-aware, persisted, no premature LRU reuse."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_revoked: dict[str, float] = {}  # jti -> exp (unix seconds)
_store_path: Path | None = None
_MAX_ENTRIES = 50_000


def configure_revocation_store(data_dir: Path) -> None:
    global _store_path
    _store_path = data_dir / ".revoked_jtis.json"
    _load()


def _load() -> None:
    global _revoked
    if _store_path is None or not _store_path.exists():
        return
    try:
        raw = json.loads(_store_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            now = time.time()
            _revoked = {str(k): float(v) for k, v in raw.items() if float(v) > now}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        logger.warning("Could not load revoked JTIs; starting fresh")


def _persist() -> None:
    if _store_path is None:
        return
    try:
        _store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(_revoked, separators=(",", ":"))
        # Atomic replace so a crash mid-write cannot leave a truncated store
        # that would drop all logout revocations on the next process start.
        tmp_path = _store_path.with_name(_store_path.name + ".tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.chmod(0o600)
        tmp_path.replace(_store_path)
    except OSError:
        logger.warning("Could not persist revoked JTIs")
        try:
            if _store_path is not None:
                _store_path.with_name(_store_path.name + ".tmp").unlink(missing_ok=True)
        except OSError:
            pass


def _prune(now: float | None = None) -> None:
    ts = now if now is not None else time.time()
    expired = [jti for jti, exp in _revoked.items() if exp <= ts]
    for jti in expired:
        del _revoked[jti]
    if len(_revoked) > _MAX_ENTRIES:
        # Prefer dropping soonest-to-expire entries that are still live only as
        # a last resort — never revive a logout by deleting a far-future exp
        # while keeping an almost-expired one. Sort by exp ascending.
        overflow = len(_revoked) - _MAX_ENTRIES
        # Drop entries closest to expiry first (they become irrelevant soonest).
        # Still incorrect under extreme abuse vs unbounded store, but avoids
        # preferentially deleting long-lived revocations.
        victims = sorted(_revoked.items(), key=lambda item: item[1])[:overflow]
        for jti, _ in victims:
            del _revoked[jti]
        logger.warning(
            "Evicted %d revoked JTIs to enforce store cap (%d); "
            "those sessions may validate until natural JWT expiry",
            overflow,
            _MAX_ENTRIES,
        )


def revoke_jti(jti: str, exp: float) -> None:
    with _LOCK:
        _revoked[str(jti)] = float(exp)
        _prune()
        _persist()


def is_jti_revoked(jti: str) -> bool:
    with _LOCK:
        _prune()
        exp = _revoked.get(str(jti))
        if exp is None:
            return False
        if exp <= time.time():
            del _revoked[str(jti)]
            return False
        return True


def clear_revocations_for_tests() -> None:
    with _LOCK:
        _revoked.clear()
