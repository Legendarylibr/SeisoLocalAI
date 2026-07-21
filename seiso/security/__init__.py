"""Path validation and subprocess sandboxing — secure by default."""

from __future__ import annotations

import os
import re
import secrets
from pathlib import Path

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
        # Reject separators / parent refs inside a single segment so callers cannot
        # smuggle ``alice/../bob`` past the exact ``"."`` / ``".."`` checks.
        if "/" in part or "\\" in part or _UNSAFE_SEGMENT.search(part):
            raise SecurityError(f"Unsafe characters in segment: {part!r}")
        if ".." in Path(part).parts:
            raise SecurityError(f"Invalid path segment: {part!r}")
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


# Mirrors forge.services.user_paths user-scoped roots (keep in sync).
USER_SCOPED_DATA_ROOTS = frozenset(
    {
        "uploads",
        "knowledge",
        "artifacts",
        "sandbox",
        "models",
        "checkpoints",
        "exports",
        "compress",
        "distill_rl",
        "rl_quant",
        "recipes",
    }
)


def assert_user_scoped_path(
    data_dir: Path,
    user_id: str,
    target: Path | str,
) -> Path:
    """Require *target* under ``data_dir/<scoped_root>/<user_id>/...``."""
    if not user_id or "/" in user_id or "\\" in user_id or user_id in {".", ".."}:
        raise SecurityError(f"Invalid user_id: {user_id!r}")
    base = Path(data_dir).expanduser().resolve()
    source = Path(target).expanduser()
    target_r = assert_within(base, source if source.is_absolute() else source.resolve())
    try:
        rel = target_r.relative_to(base)
    except ValueError as exc:
        raise SecurityError(f"Path {target_r} is outside sandbox {base}") from exc
    if len(rel.parts) < 2:
        raise SecurityError(f"Path must be under a user-scoped root for {user_id}")
    root, owner = rel.parts[0], rel.parts[1]
    if root not in USER_SCOPED_DATA_ROOTS:
        raise SecurityError(f"Access denied to path root: {root!r}")
    if owner != user_id:
        raise SecurityError(f"Path must be under {root}/{user_id}/")
    return target_r


def sanitize_filename(name: str, max_len: int = 255) -> str:
    """Produce a safe filename from user input."""
    cleaned = re.sub(r"[^\w.\- ]", "_", name.strip())
    cleaned = cleaned.strip(". ") or "unnamed"
    return cleaned[:max_len]
