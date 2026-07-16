"""Recover Seiso model inventory entries from the Hugging Face cache."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.db.store import Database
from forge.services.artifact_integrity import gguf_files_complete_with_hub
from forge.services.hf_hub import (
    estimate_snapshot_download_bytes,
    link_inventory,
)
from forge.services.hf_hub_gguf_select import (
    _inventory_name_for_files,
    list_complete_gguf_file_groups,
)
from forge.services.user_paths import user_dir
from seiso.models.catalog import CatalogEntry, get_by_gguf_mirror, get_by_repo
from seiso.models.gguf_quant import extract_quant_label_from_text
from seiso.security import sanitize_filename


def _repo_id_from_cache_dir(path: Path) -> str | None:
    name = path.name
    if not name.startswith("models--"):
        return None
    repo = name.removeprefix("models--").replace("--", "/")
    return repo if "/" in repo else None


def _catalog_entry_for_cached_repo(repo_id: str) -> CatalogEntry | None:
    return get_by_repo(repo_id) or get_by_gguf_mirror(repo_id)


def _display_name_for_shards(filename: str) -> str:
    stem = Path(filename).stem
    marker = "-00001-of-"
    if marker in stem:
        return stem.split(marker, 1)[0]
    return stem


@dataclass(frozen=True)
class _SnapshotInventory:
    gguf_files: list[str]
    weight_count: int
    weight_size: int


_SNAPSHOT_WEIGHT_SUFFIXES = (".safetensors", ".bin")


def _snapshot_inventory(snapshot_dir: Path) -> _SnapshotInventory:
    """Collect model-file facts from a snapshot with a single directory walk."""
    gguf_files: list[str] = []
    weight_count = 0
    weight_size = 0
    root = str(snapshot_dir)
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                            continue
                        if not entry.is_file():
                            continue
                        name = entry.name.lower()
                        if name.endswith(".gguf"):
                            gguf_files.append(os.path.relpath(entry.path, root))
                        elif name.endswith(_SNAPSHOT_WEIGHT_SUFFIXES):
                            weight_count += 1
                            weight_size += entry.stat().st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return _SnapshotInventory(
        gguf_files=gguf_files,
        weight_count=weight_count,
        weight_size=weight_size,
    )


def _cache_tree_mtime(hf_cache_dir: Path) -> float:
    """Fast invalidation probe — repo dirs + snapshots only, not full cache walk."""
    try:
        latest = hf_cache_dir.stat().st_mtime
    except OSError:
        return 0.0
    for child in hf_cache_dir.iterdir():
        if not child.name.startswith("models--"):
            continue
        try:
            latest = max(latest, child.stat().st_mtime)
        except OSError:
            continue
        snapshots = child / "snapshots"
        if not snapshots.is_dir():
            continue
        with contextlib.suppress(OSError):
            latest = max(latest, snapshots.stat().st_mtime)
        for snap in snapshots.iterdir():
            if not snap.is_dir():
                continue
            try:
                latest = max(latest, snap.stat().st_mtime)
            except OSError:
                continue
    return latest


def _latest_snapshot_dirs(repo_cache_dir: Path) -> list[Path]:
    snapshots_dir = repo_cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return []
    snapshots: list[tuple[float, Path]] = []
    for path in snapshots_dir.iterdir():
        if not path.is_dir():
            continue
        try:
            snapshots.append((path.stat().st_mtime, path))
        except OSError:
            continue
    return [
        path
        for _mtime, path in sorted(snapshots, key=lambda item: item[0], reverse=True)
    ]


def _gguf_inventory_source(inventory_repo: str, filenames: list[str], *, canonical: bool) -> str:
    """Unique inventory source key. Canonical preferred quant keeps ``hf:repo``."""
    if canonical:
        return f"hf:{inventory_repo}"
    return f"hf:{inventory_repo}:{Path(filenames[0]).name}"


def _gguf_records_from_snapshot(
    *,
    repo_id: str,
    snapshot_dir: Path,
    inventory: _SnapshotInventory,
    data_dir: Path,
    user_id: str,
    hf_cache_dir: Path,
) -> list[dict[str, Any]]:
    """Register every complete GGUF quant present in the cache snapshot."""
    rel_files = inventory.gguf_files
    if not rel_files:
        return []

    entry = _catalog_entry_for_cached_repo(repo_id)
    inventory_repo = entry.repo_id if entry else repo_id
    preferred = (entry.quant if entry else None) or "Q4_K_M"
    groups = list_complete_gguf_file_groups(rel_files)
    if not groups:
        return []

    def _group_rank(filenames: list[str]) -> tuple[int, str]:
        label = (extract_quant_label_from_text(filenames[0]) or "").upper()
        preferred_u = preferred.upper().replace("-", "_")
        exact = 0 if preferred_u and preferred_u in label.replace("-", "_") else 1
        return (exact, filenames[0].lower())

    groups = sorted(groups, key=_group_rank)

    inventory_dir = user_dir(data_dir, user_id, "models")
    records: list[dict[str, Any]] = []
    for index, filenames in enumerate(groups):
        paths = [snapshot_dir / filename for filename in filenames]
        if not gguf_files_complete_with_hub(
            repo_id=repo_id,
            filenames=filenames,
            paths=paths,
            entry=entry,
        ):
            continue

        target = paths[0] if len(paths) == 1 else paths[0].parent
        link = link_inventory(
            inventory_dir,
            str(_inventory_name_for_files(inventory_repo, filenames)),
            target,
        )
        size_bytes = sum(path.stat().st_size for path in paths)
        quant = extract_quant_label_from_text(filenames[0]) or (
            entry.quant if entry else None
        )
        metadata: dict[str, Any] = {
            "repo_id": inventory_repo,
            "cache_dir": str(hf_cache_dir),
            "gguf_file": filenames[0],
            "gguf_files": filenames,
        }
        if quant:
            metadata["quant"] = quant
        if repo_id != inventory_repo:
            metadata["gguf_repo"] = repo_id

        # First complete group (preferred quant when present) keeps the
        # historical hf:repo source so download/cache lookups stay stable.
        records.append(
            {
                "source": _gguf_inventory_source(
                    inventory_repo, filenames, canonical=index == 0
                ),
                "name": (
                    target.name
                    if target.is_file()
                    else _display_name_for_shards(filenames[0])
                ),
                "path": str(link.absolute()),
                "format": "gguf",
                "size_bytes": size_bytes,
                "metadata": metadata,
            }
        )
    return records


def _snapshot_record(
    *,
    repo_id: str,
    snapshot_dir: Path,
    inventory: _SnapshotInventory,
    data_dir: Path,
    user_id: str,
    hf_cache_dir: Path,
) -> dict[str, Any] | None:
    weight_count = inventory.weight_count
    weight_size = inventory.weight_size
    if weight_count <= 0:
        return None

    entry = _catalog_entry_for_cached_repo(repo_id)
    if entry:
        try:
            expected_size = estimate_snapshot_download_bytes(repo_id)
        except Exception:
            expected_size = 0
        if expected_size > 0 and weight_size < expected_size:
            return None

    inventory_repo = entry.repo_id if entry else repo_id
    inventory_dir = user_dir(data_dir, user_id, "models")
    link = link_inventory(
        inventory_dir,
        sanitize_filename(inventory_repo.replace("/", "--")),
        snapshot_dir.resolve(),
    )
    return {
        "source": f"hf:{inventory_repo}",
        "name": inventory_repo.split("/")[-1],
        "path": str(link.absolute()),
        "format": "safetensors",
        "size_bytes": weight_size,
        "metadata": {"repo_id": inventory_repo, "cache_dir": str(hf_cache_dir)},
    }


_SYNC_TTL_SEC = 30.0


@dataclass
class _SyncState:
    synced_at: float
    cache_mtime: float


_sync_states: dict[str, _SyncState] = {}


def _scan_hf_cache_records(
    *,
    hf_cache_dir: Path,
    data_dir: Path,
    user_id: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for repo_cache_dir in hf_cache_dir.iterdir():
        repo_id = _repo_id_from_cache_dir(repo_cache_dir)
        if not repo_id:
            continue
        for snapshot_dir in _latest_snapshot_dirs(repo_cache_dir):
            inventory = _snapshot_inventory(snapshot_dir)
            gguf_records = _gguf_records_from_snapshot(
                repo_id=repo_id,
                snapshot_dir=snapshot_dir,
                inventory=inventory,
                data_dir=data_dir,
                user_id=user_id,
                hf_cache_dir=hf_cache_dir,
            )
            if gguf_records:
                records.extend(gguf_records)
                break
            record = _snapshot_record(
                repo_id=repo_id,
                snapshot_dir=snapshot_dir,
                inventory=inventory,
                data_dir=data_dir,
                user_id=user_id,
                hf_cache_dir=hf_cache_dir,
            )
            if record:
                records.append(record)
                break
    return records


async def sync_hf_cache_inventory(
    db: Database,
    user_id: str,
    *,
    data_dir: Path,
    hf_cache_dir: Path,
) -> int:
    """Register completed HF cache snapshots without copying blobs."""
    if not hf_cache_dir.is_dir():
        return 0

    try:
        cache_mtime = _cache_tree_mtime(hf_cache_dir)
        cache_key = f"{user_id}:{hf_cache_dir.resolve()}"
    except OSError:
        cache_mtime = 0.0
        cache_key = f"{user_id}:{hf_cache_dir}"

    now = time.monotonic()
    state = _sync_states.get(cache_key)
    if (
        state is not None
        and now - state.synced_at < _SYNC_TTL_SEC
        and state.cache_mtime == cache_mtime
    ):
        return 0

    records = await asyncio.to_thread(
        _scan_hf_cache_records,
        hf_cache_dir=hf_cache_dir,
        data_dir=data_dir,
        user_id=user_id,
    )
    registered = await db.upsert_models(user_id, records)
    _sync_states[cache_key] = _SyncState(synced_at=now, cache_mtime=cache_mtime)
    return registered
