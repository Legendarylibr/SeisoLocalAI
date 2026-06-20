"""Shared helpers for vendored pipeline config builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_preset(
    presets: dict[str, dict[str, Any]],
    preset_name: str,
    *,
    default: str = "smoke",
) -> tuple[str, dict[str, Any]]:
    """Return (canonical_name, preset_dict) with fallback to default."""
    name = str(preset_name or default)
    return name, dict(presets.get(name, presets[default]))


def job_output_root(data_dir: Path, pipeline: str, user_id: str, job_id: str) -> Path:
    """Create and return the per-job output directory under the user sandbox."""
    root = data_dir / pipeline / user_id / job_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def validate_stages(stages: list[str], stage_order: tuple[str, ...]) -> None:
    """Raise ValueError when a stage is not in the pipeline's allowed order."""
    allowed = set(stage_order)
    for stage in stages:
        if stage not in allowed:
            raise ValueError(f"Unknown pipeline stage: {stage}")


def resolve_config_file_path(config_file: str | None, *, vendor_root: Path) -> Path | None:
    """Resolve a config file path from an absolute path or vendor configs dir."""
    if not config_file:
        return None
    path = Path(config_file)
    if path.is_file():
        return path
    vendor_path = vendor_root / "configs" / config_file
    return vendor_path if vendor_path.is_file() else None
