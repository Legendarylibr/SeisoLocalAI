"""Recover Seiso model inventory entries from the Hugging Face cache."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.db.store import Database
from forge.services.hf_hub import (
    _inventory_name_for_files,
    _pick_gguf_files,
    dir_size,
    link_inventory,
)
from forge.services.user_paths import user_dir
from seiso.models.catalog import CATALOG, CatalogEntry, get_by_gguf_mirror
from seiso.security import sanitize_filename


def _repo_id_from_cache_dir(path: Path) -> str | None:
    name = path.name
    if not name.startswith("models--"):
        return None
    repo = name.removeprefix("models--").replace("--", "/")
    return repo if "/" in repo else None


def _catalog_entry_for_cached_repo(repo_id: str) -> CatalogEntry | None:
    direct = next((entry for entry in CATALOG if entry.repo_id == repo_id), None)
    if direct:
        return direct
    mirror = get_by_gguf_mirror(repo_id)
    if mirror:
        return mirror
    return next((entry for entry in CATALOG if entry.gguf_repo == repo_id), None)


def _display_name_for_shards(filename: str) -> str:
    stem = Path(filename).stem
    marker = "-00001-of-"
    if marker in stem:
        return stem.split(marker, 1)[0]
    return stem


def _latest_snapshot_dirs(repo_cache_dir: Path) -> list[Path]:
    snapshots_dir = repo_cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return []
    return sorted(
        (p for p in snapshots_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _gguf_record_from_snapshot(
    *,
    repo_id: str,
    snapshot_dir: Path,
    data_dir: Path,
    user_id: str,
    hf_cache_dir: Path,
) -> dict[str, Any] | None:
    rel_files = [
        str(path.relative_to(snapshot_dir))
        for path in snapshot_dir.rglob("*.gguf")
        if path.is_file()
    ]
    if not rel_files:
        return None

    entry = _catalog_entry_for_cached_repo(repo_id)
    inventory_repo = entry.repo_id if entry else repo_id
    quant = entry.quant if entry else "Q4_K_M"
    filenames = _pick_gguf_files(rel_files, preferred_quant=quant, repo_id=inventory_repo)
    if not filenames:
        return None

    paths = [snapshot_dir / filename for filename in filenames]
    if not all(path.is_file() for path in paths):
        return None

    target = paths[0] if len(paths) == 1 else paths[0].parent
    inventory_dir = user_dir(data_dir, user_id, "models")
    link = link_inventory(
        inventory_dir,
        str(_inventory_name_for_files(inventory_repo, filenames)),
        target,
    )
    size_bytes = sum(path.stat().st_size for path in paths)
    metadata: dict[str, Any] = {
        "repo_id": inventory_repo,
        "cache_dir": str(hf_cache_dir),
        "gguf_file": filenames[0],
        "gguf_files": filenames,
    }
    if repo_id != inventory_repo:
        metadata["gguf_repo"] = repo_id

    return {
        "source": f"hf:{inventory_repo}",
        "name": target.name if target.is_file() else _display_name_for_shards(filenames[0]),
        "path": str(link.absolute()),
        "format": "gguf",
        "size_bytes": size_bytes,
        "metadata": metadata,
    }


def _snapshot_record(
    *,
    repo_id: str,
    snapshot_dir: Path,
    data_dir: Path,
    user_id: str,
    hf_cache_dir: Path,
) -> dict[str, Any] | None:
    weight_files = [
        path
        for path in snapshot_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".safetensors", ".bin"}
    ]
    if not weight_files:
        return None

    entry = _catalog_entry_for_cached_repo(repo_id)
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
        "size_bytes": dir_size(snapshot_dir),
        "metadata": {"repo_id": inventory_repo, "cache_dir": str(hf_cache_dir)},
    }


_SYNC_TTL_SEC = 30.0


@dataclass
class _SyncState:
    synced_at: float
    cache_mtime: float


_sync_states: dict[str, _SyncState] = {}


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
        cache_mtime = hf_cache_dir.stat().st_mtime
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

    registered = 0
    for repo_cache_dir in hf_cache_dir.iterdir():
        repo_id = _repo_id_from_cache_dir(repo_cache_dir)
        if not repo_id:
            continue
        for snapshot_dir in _latest_snapshot_dirs(repo_cache_dir):
            record = _gguf_record_from_snapshot(
                repo_id=repo_id,
                snapshot_dir=snapshot_dir,
                data_dir=data_dir,
                user_id=user_id,
                hf_cache_dir=hf_cache_dir,
            ) or _snapshot_record(
                repo_id=repo_id,
                snapshot_dir=snapshot_dir,
                data_dir=data_dir,
                user_id=user_id,
                hf_cache_dir=hf_cache_dir,
            )
            if not record:
                continue
            await db.upsert_model(
                user_id=user_id,
                source=record.pop("source"),
                **record,
            )
            registered += 1
            break
    _sync_states[cache_key] = _SyncState(synced_at=now, cache_mtime=cache_mtime)
    return registered
