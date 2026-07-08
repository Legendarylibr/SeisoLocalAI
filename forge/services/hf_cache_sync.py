"""Hugging Face cache inventory background sync scheduling."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from forge.db.store import Database
from forge.services.hf_cache_inventory import sync_hf_cache_inventory
from forge.services.job_runtime import spawn_background

logger = logging.getLogger(__name__)

_MODEL_CACHE_BACKGROUND_SYNC_TTL_S = 120.0
_model_cache_background_syncs: dict[str, float] = {}


async def sync_hf_cache_inventory_background(
    db: Database,
    user_id: str,
    *,
    data_dir: Path,
    hf_cache_dir: Path,
) -> None:
    try:
        await sync_hf_cache_inventory(
            db,
            user_id,
            data_dir=data_dir,
            hf_cache_dir=hf_cache_dir,
        )
    except Exception:
        logger.exception("Background Hugging Face cache inventory sync failed")


async def schedule_hf_cache_inventory_sync(
    db: Database,
    user_id: str,
    *,
    data_dir: Path,
    hf_cache_dir: Path,
    sync_cache: bool,
) -> None:
    if sync_cache:
        await sync_hf_cache_inventory(
            db,
            user_id,
            data_dir=data_dir,
            hf_cache_dir=hf_cache_dir,
        )
        return

    cache_key = f"{user_id}:{hf_cache_dir}"
    now = time.monotonic()
    last_sync = _model_cache_background_syncs.get(cache_key, 0.0)
    if now - last_sync >= _MODEL_CACHE_BACKGROUND_SYNC_TTL_S:
        _model_cache_background_syncs[cache_key] = now
        spawn_background(
            sync_hf_cache_inventory_background(
                db,
                user_id,
                data_dir=data_dir,
                hf_cache_dir=hf_cache_dir,
            )
        )
