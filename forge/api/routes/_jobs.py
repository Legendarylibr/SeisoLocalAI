"""Shared helpers for pipeline job API routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from forge.api.http_errors import raise_forbidden
from seiso.security import SecurityError


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


def validate_pipeline_paths(
    data_dir: Path,
    user_id: str,
    config: dict[str, Any],
    *,
    config_file: str | None = None,
    path_keys: tuple[str, ...] = (),
    llama_cpp_binary: bool = False,
) -> None:
    """Validate user-scoped paths; raise HTTP 403 on sandbox violations."""
    from forge.services.user_paths import (
        assert_llama_cpp_binary,
        assert_user_config_file,
        assert_user_path,
    )

    try:
        if config_file:
            assert_user_config_file(data_dir, user_id, config_file)
        for key in path_keys:
            if config.get(key):
                assert_user_path(data_dir, user_id, config[key])
        if llama_cpp_binary and config.get("llama_cpp_binary"):
            assert_llama_cpp_binary(config["llama_cpp_binary"])
    except SecurityError as exc:
        raise_forbidden(exc)


def enrich_stage_results(result: dict[str, Any], *extra_keys: str) -> dict[str, Any]:
    """Merge manifest and optional artifact keys into stage_results."""
    stage_results = dict(result.get("stage_results") or {})
    for key in ("manifest", *extra_keys):
        if value := result.get(key):
            stage_results[key] = value
    return stage_results
