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
        _store_path.write_text(
            json.dumps(_revoked, separators=(",", ":")), encoding="utf-8"
        )
        _store_path.chmod(0o600)
    except OSError:
        logger.warning("Could not persist revoked JTIs")


def _prune(now: float | None = None) -> None:
    ts = now if now is not None else time.time()
    expired = [jti for jti, exp in _revoked.items() if exp <= ts]
    for jti in expired:
        del _revoked[jti]
    if len(_revoked) > _MAX_ENTRIES:
        overflow = len(_revoked) - _MAX_ENTRIES
        for jti, _ in sorted(_revoked.items(), key=lambda item: item[1])[:overflow]:
            del _revoked[jti]
        logger.warning(
            "Evicted %d revoked JTIs to enforce store cap (%d)",
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
