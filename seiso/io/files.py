"""Filesystem helpers shared across Seiso and Forge."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Iterator, Protocol

MODEL_WEIGHT_SUFFIXES = frozenset({".gguf", ".safetensors", ".bin", ".pt", ".pth"})


class _StatEntry(Protocol):
    name: str
    path: str

    def stat(self) -> os.stat_result: ...


def _iter_file_entries(path: Path, pattern: str = "*") -> Iterator[_StatEntry]:
    """Yield matching files under *path* with minimal Path allocations."""
    if path.is_file():
        if fnmatch.fnmatchcase(path.name, pattern):
            yield _PathStatEntry(path)
        return
    stack = [str(path)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file() and fnmatch.fnmatchcase(
                            entry.name, pattern
                        ):
                            yield entry
                    except OSError:
                        continue
        except OSError:
            continue


class _PathStatEntry:
    """Small adapter for the rare direct-file path in _iter_file_entries."""

    def __init__(self, path: Path) -> None:
        self.name = path.name
        self.path = str(path)
        self._path = path

    def stat(self) -> os.stat_result:
        return self._path.stat()


def path_size_bytes(path: Path) -> int:
    """Return byte size for a file or recursive byte size for a directory."""
    if path.is_file():
        return path.stat().st_size
    return sum(entry.stat().st_size for entry in _iter_file_entries(path))


def iter_matching_files(
    path: Path,
    pattern: str = "*",
    *,
    suffixes: set[str] | frozenset[str] | None = None,
) -> Iterator[Path]:
    """Yield matching files below *path* without allocating Paths for non-matches."""
    normalized_suffixes = (
        {suffix.lower() for suffix in suffixes} if suffixes is not None else None
    )
    for entry in _iter_file_entries(path, pattern):
        if (
            normalized_suffixes is not None
            and os.path.splitext(entry.name)[1].lower() not in normalized_suffixes
        ):
            continue
        yield Path(entry.path)


def model_weight_size_bytes(path: Path) -> int:
    """Return weight-file bytes for a model path, falling back to full size if needed."""
    if path.is_file():
        return path.stat().st_size
    count, total = matching_file_stats(path, suffixes=MODEL_WEIGHT_SUFFIXES)
    if count > 0:
        return total
    return path_size_bytes(path)


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
    for entry in _iter_file_entries(path, pattern):
        if (
            normalized_suffixes is not None
            and os.path.splitext(entry.name)[1].lower() not in normalized_suffixes
        ):
            continue
        count += 1
        total_size += entry.stat().st_size
    return count, total_size
