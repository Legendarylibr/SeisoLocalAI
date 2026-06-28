"""Register training/export outputs into the local model inventory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge.db.store import Database
from forge.services.inference_models import backend_for_path
from forge.services.user_paths import assert_user_path
from seiso.security import sanitize_filename


async def register_model_path(
    db: Database,
    *,
    user_id: str,
    data_dir: Path,
    path: str | Path,
    name: str,
    source: str,
    model_format: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict | None:
    """Add a filesystem model to inventory if it exists and is not already registered."""
    try:
        resolved = assert_user_path(data_dir, user_id, str(path))
    except Exception:
        return None
    if not resolved.exists():
        return None

    fmt = model_format or (
        resolved.suffix.lstrip(".") if resolved.is_file() else "safetensors"
    )
    size = (
        resolved.stat().st_size
        if resolved.is_file()
        else sum(f.stat().st_size for f in resolved.rglob("*") if f.is_file())
    )

    existing = await db.list_models(user_id)
    norm = str(resolved.resolve())
    if any(str(Path(m["path"]).resolve()) == norm for m in existing):
        return None

    meta = dict(metadata or {})
    meta["default_backend"] = backend_for_path(norm, fmt)

    return await db.add_model(
        user_id=user_id,
        name=sanitize_filename(name),
        path=str(resolved),
        source=source,
        format=fmt,
        size_bytes=size,
        metadata=meta,
    )


async def register_training_checkpoint(
    db: Database,
    *,
    user_id: str,
    data_dir: Path,
    checkpoint_path: str,
    job_id: str,
) -> dict | None:
    try:
        resolved = assert_user_path(data_dir, user_id, checkpoint_path)
    except Exception:
        return None
    if not resolved.exists():
        return None

    size = sum(f.stat().st_size for f in resolved.rglob("*") if f.is_file())
    meta = {"job_id": job_id, "origin": "fine-tune"}
    norm = str(resolved.resolve())
    meta["default_backend"] = backend_for_path(norm, "safetensors")

    return await db.upsert_model(
        user_id,
        "training",
        name=sanitize_filename(f"checkpoint-{job_id[:8]}"),
        path=str(resolved),
        format="safetensors",
        size_bytes=size,
        metadata=meta,
    )


async def register_export_outputs(
    db: Database,
    *,
    user_id: str,
    data_dir: Path,
    outputs: dict[str, str],
    job_id: str,
) -> list[dict]:
    registered: list[dict] = []
    for key, raw_path in outputs.items():
        path = Path(raw_path)
        if "gguf" in key.lower() or path.suffix.lower() == ".gguf":
            fmt = "gguf"
            name = path.parent.name if path.is_file() else path.name
            meta = {"job_id": job_id, "export_key": key}
            modelfile = (
                path.parent / "Modelfile" if path.is_file() else path / "Modelfile"
            )
            if modelfile.is_file():
                meta["modelfile"] = str(modelfile)
            entry = await register_model_path(
                db,
                user_id=user_id,
                data_dir=data_dir,
                path=path if path.is_dir() else path.parent,
                name=name,
                source="export",
                model_format=fmt,
                metadata=meta,
            )
        elif key in {"merged", "lora", "full", "base"} or path.is_dir():
            entry = await register_model_path(
                db,
                user_id=user_id,
                data_dir=data_dir,
                path=path,
                name=f"{key}-{job_id[:8]}",
                source="export",
                model_format="safetensors",
                metadata={"job_id": job_id, "export_key": key},
            )
        else:
            entry = None
        if entry:
            registered.append(entry)
    return registered
