"""Shared checks that local model artifacts are fully downloaded."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from seiso.inference.backends import gguf_is_supported_by_llamacpp
from seiso.models.catalog import CatalogEntry


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
        ggufs = [p for p in path.rglob("*.gguf") if p.is_file()]
        size = sum(p.stat().st_size for p in ggufs)
        return bool(ggufs) and size > 0 and (expected_size <= 0 or size >= expected_size)
    if path.is_dir():
        weight_files = [
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in {".safetensors", ".bin"}
        ]
        size = sum(p.stat().st_size for p in weight_files)
        return bool(weight_files) and size > 0 and (expected_size <= 0 or size >= expected_size)
    return (
        path.is_file()
        and path.stat().st_size > 0
        and (expected_size <= 0 or path.stat().st_size >= expected_size)
    )


def gguf_files_complete_at_path(path: Path, filenames: list[str], expected_size: int) -> bool:
    if not filenames:
        return False
    files = (
        [path]
        if path.is_file() and len(filenames) == 1
        else [path / filename for filename in filenames]
    )
    if not all(item.is_file() and item.stat().st_size > 0 for item in files):
        return False
    actual_size = sum(item.stat().st_size for item in files)
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
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        return False
    if size_lookup is None:
        from forge.services.hf_hub import get_gguf_file_size_bytes

        size_lookup = get_gguf_file_size_bytes
    try:
        expected = sum(size_lookup(repo_id, filename) for filename in filenames)
    except Exception:
        return entry is None
    actual = sum(path.stat().st_size for path in paths)
    return expected <= 0 or actual >= expected


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

    local_files = [path] if path.is_file() else [path / str(filename) for filename in gguf_files]
    if not all(item.is_file() for item in local_files):
        return False
    actual_size = sum(item.stat().st_size for item in local_files)
    if size_lookup is None:
        from forge.services.hf_hub import get_gguf_file_size_bytes

        size_lookup = get_gguf_file_size_bytes
    try:
        expected_size = sum(size_lookup(gguf_repo, str(filename)) for filename in gguf_files)
    except Exception:
        return True
    return expected_size <= 0 or actual_size >= expected_size
