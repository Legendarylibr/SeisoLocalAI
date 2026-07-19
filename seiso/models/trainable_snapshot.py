"""Detect Hugging Face snapshots suitable for LoRA/SFT training."""

from __future__ import annotations

import re
from pathlib import Path

from seiso.io.files import iter_matching_files

_TRAINABLE_WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin"})
_GGUF_REPO_RE = re.compile(r"(?:^|/)[^/]*-gguf(?:$|/|-)", re.I)

GGUF_ONLY_REPO_MESSAGE = (
    "This model cache is GGUF-only (chat/inference weights). LoRA/QLoRA training needs a "
    "safetensors or PyTorch checkpoint — pick a non-GGUF Hugging Face repo or download "
    "the safetensors variant from the catalog."
)


def is_gguf_only_repo_id(
    repo_id: str, tags: list[str] | tuple[str, ...] | None = None
) -> bool:
    """True when *repo_id* points at a GGUF inference mirror, not a trainable checkpoint."""
    lowered = repo_id.strip().lower()
    if not lowered:
        return False
    if _GGUF_REPO_RE.search(lowered) or lowered.endswith("-gguf"):
        return True
    tag_set = {t.lower() for t in (tags or ())}
    return bool(
        "gguf" in tag_set and "safetensors" not in tag_set and "pytorch" not in tag_set
    )


def snapshot_has_trainable_weights(root: Path) -> bool:
    """True when *root* contains at least one safetensors/bin weight file."""
    path = root.expanduser()
    if path.is_file():
        return path.suffix.lower() in _TRAINABLE_WEIGHT_SUFFIXES
    return any(iter_matching_files(path, suffixes=_TRAINABLE_WEIGHT_SUFFIXES))
