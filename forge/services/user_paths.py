"""Per-user filesystem path policy — tenant isolation under shared data_dir."""

from __future__ import annotations

from pathlib import Path

from seiso.security import SecurityError, assert_within, safe_join

_USER_SCOPED_ROOTS = frozenset({"uploads", "knowledge", "artifacts", "sandbox", "models", "checkpoints", "exports"})


def user_dir(sandbox_root: Path, user_id: str, category: str) -> Path:
    """Return (and does not create) a user-owned directory under category."""
    if category not in _USER_SCOPED_ROOTS:
        raise SecurityError(f"Unknown user path category: {category}")
    return safe_join(sandbox_root, category, user_id)


def assert_user_path(sandbox_root: Path, user_id: str, target: str | Path) -> Path:
    """Path must be inside sandbox and under an allowed root scoped to user_id."""
    source = assert_within(sandbox_root, Path(target).expanduser())
    rel = source.relative_to(sandbox_root.resolve())
    if not rel.parts:
        raise SecurityError("Invalid path")
    root = rel.parts[0]
    if root not in _USER_SCOPED_ROOTS:
        raise SecurityError(f"Access denied to path root: {root!r}")
    if len(rel.parts) < 2 or rel.parts[1] != user_id:
        raise SecurityError(f"Path must be under {root}/{user_id}/")
    return source
