"""Shared helpers for bundled pipeline config builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_preset(
    presets: dict[str, dict[str, Any]],
    preset_name: str,
    *,
    default: str = "smoke",
) -> tuple[str, dict[str, Any]]:
    """Return (canonical_name, preset_dict), rejecting unknown preset names."""
    name = str(preset_name or default)
    if name not in presets:
        known = ", ".join(sorted(presets))
        raise ValueError(f"Unknown preset: {name}. Choose one of: {known}")
    return name, dict(presets[name])


def job_output_root(data_dir: Path, pipeline: str, user_id: str, job_id: str) -> Path:
    """Create and return the per-job output directory under the user sandbox."""
    from seiso.security import safe_join

    root = safe_join(data_dir, pipeline, user_id, job_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def validate_stages(stages: list[str], stage_order: tuple[str, ...]) -> None:
    """Raise ValueError when a stage is not in the pipeline's allowed order."""
    allowed = set(stage_order)
    for stage in stages:
        if stage not in allowed:
            raise ValueError(f"Unknown pipeline stage: {stage}")


def sort_stages(stages: list[str], stage_order: tuple[str, ...]) -> list[str]:
    """Return stages sorted by canonical pipeline order (membership already validated)."""
    order_index = {name: idx for idx, name in enumerate(stage_order)}
    return sorted(stages, key=lambda stage: order_index[stage])


def resolve_config_file_path(config_file: str | None, *, bundle_root: Path) -> Path | None:
    """Resolve a config file path from an absolute path or bundled configs dir."""
    if not config_file:
        return None
    path = Path(config_file)
    if path.is_file():
        return path
    bundled_path = bundle_root / "configs" / config_file
    return bundled_path if bundled_path.is_file() else None
