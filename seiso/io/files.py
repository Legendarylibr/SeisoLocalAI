"""Filesystem helpers shared across Seiso and Forge."""

from __future__ import annotations

from pathlib import Path


def path_size_bytes(path: Path) -> int:
    """Return byte size for a file or recursive byte size for a directory."""
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def matching_file_stats(
    path: Path,
    pattern: str = "*",
    *,
    suffixes: set[str] | frozenset[str] | None = None,
) -> tuple[int, int]:
    """Return (file_count, total_bytes) for files matching a directory scan."""
    count = 0
    total_size = 0
    normalized_suffixes = (
        {suffix.lower() for suffix in suffixes} if suffixes is not None else None
    )
    for item in path.rglob(pattern):
        if not item.is_file():
            continue
        if (
            normalized_suffixes is not None
            and item.suffix.lower() not in normalized_suffixes
        ):
            continue
        count += 1
        total_size += item.stat().st_size
    return count, total_size
