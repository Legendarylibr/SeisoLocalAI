"""Knowledge base path policy and ID validation."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import HTTPException

from seiso.security import SecurityError, assert_within, safe_join

_KB_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def validate_kb_id(kb_id: str) -> str:
    if not _KB_ID_RE.match(kb_id):
        raise HTTPException(
            400,
            "knowledge_base_id must be 1–64 alphanumeric characters, hyphens, or underscores",
        )
    return kb_id


def assert_ingest_source(
    sandbox_root: Path, user_id: str, source_path: str | Path
) -> Path:
    """Allow ingest only from caller uploads or own knowledge dir — not other users' KB."""
    source = assert_within(sandbox_root, Path(source_path))
    rel = source.relative_to(sandbox_root.resolve())
    parts = rel.parts

    if parts and parts[0] == "knowledge":
        if len(parts) < 2 or parts[1] != user_id:
            raise SecurityError("Cannot ingest from another user's knowledge base")
        return source

    uploads_root = safe_join(sandbox_root, "uploads", user_id)
    try:
        source.relative_to(uploads_root.resolve())
        return source
    except ValueError as exc:
        raise SecurityError(
            f"Ingest source must be under uploads/{user_id}/ or your own knowledge base"
        ) from exc
