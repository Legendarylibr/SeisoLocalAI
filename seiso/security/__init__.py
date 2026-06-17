"""Path validation and subprocess sandboxing — secure by default."""

from __future__ import annotations

import os
import re
import secrets
from pathlib import Path
from typing import Iterable

# Characters forbidden in user-supplied path segments
_UNSAFE_SEGMENT = re.compile(r"[\x00<>:\"|?*]")


class SecurityError(PermissionError):
    """Raised when a path or operation violates sandbox policy."""


def generate_secret_key(nbytes: int = 32) -> str:
    return secrets.token_urlsafe(nbytes)


def resolve_data_dir(raw: str | Path | None = None) -> Path:
    """Resolve and create the Seiso data directory."""
    if raw is None:
        raw = os.environ.get("SEISO_DATA_DIR", "~/.seiso")
    path = Path(os.path.expanduser(str(raw))).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_within(base: Path, target: Path) -> bool:
    """True if target resolves inside base (no prefix tricks)."""
    base_r = base.resolve()
    target_r = target.resolve()
    try:
        target_r.relative_to(base_r)
        return True
    except ValueError:
        return False


def safe_join(base: Path, *parts: str) -> Path:
    """Join paths ensuring the result stays within base (no traversal)."""
    base = base.resolve()
    candidate = base
    for part in parts:
        if not part or part in (".", ".."):
            raise SecurityError(f"Invalid path segment: {part!r}")
        if _UNSAFE_SEGMENT.search(part):
            raise SecurityError(f"Unsafe characters in segment: {part!r}")
        candidate = (candidate / part).resolve()
        if not _is_within(base, candidate):
            raise SecurityError("Path traversal detected")
    return candidate


def assert_within(base: Path, target: Path) -> Path:
    """Verify target is inside base; return resolved target."""
    base_r = base.resolve()
    target_r = target.resolve()
    if not _is_within(base_r, target_r):
        raise SecurityError(f"Path {target_r} is outside sandbox {base_r}")
    return target_r


def sanitize_filename(name: str, max_len: int = 255) -> str:
    """Produce a safe filename from user input."""
    cleaned = re.sub(r"[^\w.\- ]", "_", name.strip())
    cleaned = cleaned.strip(". ") or "unnamed"
    return cleaned[:max_len]


def allowed_extensions(exts: Iterable[str], path: Path) -> bool:
    return path.suffix.lower() in {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
