"""Shared checks that local model artifacts are fully downloaded."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from seiso.inference.backends import gguf_is_supported_by_llamacpp
from seiso.io.files import matching_file_stats
from seiso.models.catalog import CatalogEntry


def _existing_files_total_size(
    paths: list[Path], *, require_positive: bool
) -> int | None:
    total_size = 0
    for path in paths:
        if not path.is_file():
            return None
        size = path.stat().st_size
        if require_positive and size <= 0:
            return None
        total_size += size
    return total_size


def path_has_complete_artifact(path: Path, fmt: str, expected_size: int) -> bool:
    if not path.exists():
        return False
    if path.name.endswith((".incomplete", ".partial", ".lock")):
        return False
    if fmt == "gguf":
        if path.is_file():
            size = path.stat().st_size
            return (
                path.suffix.lower() == ".gguf"
                and size > 0
                and (expected_size <= 0 or size >= expected_size)
            )
        count, size = matching_file_stats(path, "*.gguf")
        return (
            count > 0 and size > 0 and (expected_size <= 0 or size >= expected_size)
        )
    if path.is_dir():
        count, size = matching_file_stats(
            path, suffixes=frozenset({".safetensors", ".bin"})
        )
        return (
            count > 0
            and size > 0
            and (expected_size <= 0 or size >= expected_size)
        )
    if not path.is_file():
        return False
    size = path.stat().st_size
    return (
        size > 0
        and (expected_size <= 0 or size >= expected_size)
    )


def gguf_files_complete_at_path(
    path: Path, filenames: list[str], expected_size: int
) -> bool:
    if not filenames:
        return False
    files = (
        [path]
        if path.is_file() and len(filenames) == 1
        else [path / filename for filename in filenames]
    )
    actual_size = _existing_files_total_size(files, require_positive=True)
    if actual_size is None:
        return False
    return expected_size <= 0 or actual_size >= expected_size


def gguf_files_complete_with_hub(
    *,
    repo_id: str,
    filenames: list[str],
    paths: list[Path],
    entry: CatalogEntry | None = None,
    size_lookup: Callable[[str, str], int] | None = None,
) -> bool:
    if not filenames or len(filenames) != len(paths):
        return False
    actual_size = _existing_files_total_size(paths, require_positive=True)
    if actual_size is None:
        return False
    if size_lookup is None:
        from forge.services.hf_hub import get_gguf_file_size_bytes

        size_lookup = get_gguf_file_size_bytes
    try:
        expected = sum(size_lookup(repo_id, filename) for filename in filenames)
    except Exception:
        return entry is None
    return expected <= 0 or actual_size >= expected


def inventory_gguf_is_complete(
    row: dict[str, Any],
    metadata: dict[str, Any],
    *,
    size_lookup: Callable[[str, str], int] | None = None,
) -> bool:
    path = Path(str(row.get("path") or ""))
    if not path.exists():
        return False
    fmt = str(row.get("format") or "").lower()
    if fmt != "gguf":
        return True
    if not gguf_is_supported_by_llamacpp(str(path)):
        return False

    gguf_repo = str(metadata.get("gguf_repo") or metadata.get("repo_id") or "")
    gguf_files = metadata.get("gguf_files") or metadata.get("gguf_file")
    if isinstance(gguf_files, str):
        gguf_files = [gguf_files]
    if not gguf_repo or not isinstance(gguf_files, list) or not gguf_files:
        return not str(row.get("source") or "").startswith("hf:")

    local_files = (
        [path] if path.is_file() else [path / str(filename) for filename in gguf_files]
    )
    actual_size = _existing_files_total_size(local_files, require_positive=False)
    if actual_size is None:
        return False
    if size_lookup is None:
        from forge.services.hf_hub import get_gguf_file_size_bytes

        size_lookup = get_gguf_file_size_bytes
    try:
        expected_size = sum(
            size_lookup(gguf_repo, str(filename)) for filename in gguf_files
        )
    except Exception:
        return True
    return expected_size <= 0 or actual_size >= expected_size
