"""Shared helpers for pipeline job API routes."""

from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException


def _parse_json_field(row: dict, key: str, fallback: Any) -> Any:
    raw = row.get(key)
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def format_stage_pipeline_job(row: dict) -> dict:
    """Decode stages_json / stage_results_json on a compress job row."""
    out = dict(row)
    out["stages"] = _parse_json_field(row, "stages_json", [])
    out["stage_results"] = _parse_json_field(row, "stage_results_json", {})
    return out


def format_rl_quant_job(row: dict) -> dict:
    """Decode gguf_quants_json / recommendation_json on an RL quant row."""
    out = dict(row)
    out["gguf_quants"] = _parse_json_field(row, "gguf_quants_json", [])
    out["recommendation"] = _parse_json_field(row, "recommendation_json", {})
    return out


def stage_presets_response(
    presets: dict[str, dict[str, Any]],
    stage_order: tuple[str, ...] | list[str],
    help: dict[str, str],
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "presets": [
            {
                "id": name,
                "label": name.replace("_", " ").title(),
                "stages": preset.get("stages", []),
            }
            for name, preset in presets.items()
        ],
        "stages": list(stage_order),
        "help": help,
    }
    if defaults:
        out["defaults"] = defaults
    return out


def apply_linked_training_job(
    config: dict[str, Any],
    train_job: dict[str, Any],
    *,
    path_key: str,
    preset_when: str | None = None,
    preset_override: str | None = None,
) -> None:
    """Copy a training checkpoint into a pipeline config when present."""
    checkpoint = train_job.get("checkpoint_path")
    if not checkpoint:
        return
    config[path_key] = checkpoint
    if preset_when and preset_override and config.get("preset") == preset_when:
        config["preset"] = preset_override


async def resolve_linked_training_job(
    db: Any,
    user_id: str,
    link_job_id: str,
    config: dict[str, Any],
    *,
    path_key: str,
    preset_when: str | None = None,
    preset_override: str | None = None,
) -> None:
    """Fetch a training job and copy its checkpoint into a pipeline config."""
    train_job = await db.get_training_job(link_job_id, user_id)
    if not train_job:
        raise HTTPException(404, "Linked training job not found")
    apply_linked_training_job(
        config,
        train_job,
        path_key=path_key,
        preset_when=preset_when,
        preset_override=preset_override,
    )
