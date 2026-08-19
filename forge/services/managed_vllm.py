"""Forge-facing managed multi-GPU vLLM helpers (optional path)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from forge.db.store import Database
from seiso.inference.managed_vllm import (
    MANAGED_PROVIDER_MARKER,
    build_launch_command,
    get_status,
    managed_vllm_enabled,
    maybe_autostart_from_env,
    provider_config_from_status,
    start_managed_vllm,
    stop_managed_vllm,
    suggest_tensor_parallel,
)

logger = logging.getLogger(__name__)

MANAGED_PROVIDER_NAME = "Managed multi-GPU vLLM"


async def ensure_managed_provider_row(
    db: Database,
    user_id: str,
    status: dict[str, Any],
) -> dict[str, Any] | None:
    """Create or refresh a local_chat provider row pointing at the managed server."""
    from forge.providers.router import LOCAL_PROVIDER_TYPES, PROVIDER_LOCAL_CHAT

    if not status.get("running"):
        return None
    config = provider_config_from_status(status)
    rows = await db.list_providers(user_id)
    for row in rows:
        if row["provider_type"].lower() not in LOCAL_PROVIDER_TYPES:
            continue
        try:
            existing = json.loads(row["config_json"])
        except (TypeError, json.JSONDecodeError):
            continue
        if existing.get("managed_by") != MANAGED_PROVIDER_MARKER:
            continue
        # Replace config by delete+create (no update API).
        await db.delete_provider(row["id"], user_id)
        break
    created = await db.create_provider(user_id, MANAGED_PROVIDER_NAME, PROVIDER_LOCAL_CHAT, config)
    created["config"] = config
    return created


async def remove_managed_provider_rows(db: Database, user_id: str) -> int:
    from forge.providers.router import LOCAL_PROVIDER_TYPES

    removed = 0
    rows = await db.list_providers(user_id)
    for row in rows:
        if row["provider_type"].lower() not in LOCAL_PROVIDER_TYPES:
            continue
        try:
            existing = json.loads(row["config_json"])
        except (TypeError, json.JSONDecodeError):
            continue
        if existing.get("managed_by") != MANAGED_PROVIDER_MARKER:
            continue
        if await db.delete_provider(row["id"], user_id):
            removed += 1
    return removed


def stop_managed_if_running(*, data_dir: Path, reason: str = "free_memory") -> dict[str, Any]:
    """Stop managed vLLM for Free memory / GPU task prep. Safe no-op when idle."""
    status = get_status()
    if not status.get("running") or not status.get("managed"):
        return {"stopped": False, "reason": "not_running"}
    logger.info("Stopping managed multi-GPU vLLM (%s)", reason)
    return stop_managed_vllm(data_dir=data_dir, force=False)


def launch_preview(
    *,
    model: str,
    tensor_parallel_size: int | None = None,
    host: str | None = None,
    port: int | None = None,
    cuda_visible_devices: str | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
) -> dict[str, Any]:
    return build_launch_command(
        model=model,
        host=host or "127.0.0.1",
        port=port or 8000,
        tensor_parallel_size=tensor_parallel_size,
        cuda_visible_devices=cuda_visible_devices,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )


__all__ = [
    "MANAGED_PROVIDER_NAME",
    "ensure_managed_provider_row",
    "get_status",
    "launch_preview",
    "managed_vllm_enabled",
    "maybe_autostart_from_env",
    "remove_managed_provider_rows",
    "start_managed_vllm",
    "stop_managed_if_running",
    "stop_managed_vllm",
    "suggest_tensor_parallel",
]
