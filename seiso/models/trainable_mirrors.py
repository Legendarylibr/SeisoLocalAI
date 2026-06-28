"""Resolve trainable Hub mirrors when official repos are gated."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from seiso.models.hub_errors import format_hub_error, is_gated_hub_error

logger = logging.getLogger(__name__)

# Official id -> open trainable mirror (same architecture, safetensors for QLoRA).
TRAINABLE_HUB_MIRRORS: dict[str, str] = {
    "google/gemma-3-12b-it": "unsloth/gemma-3-12b-it",
    "google/gemma-3-12b-pt": "unsloth/gemma-3-12b-it",
    "google/gemma-3-27b-it": "unsloth/gemma-3-27b-it",
    "google/gemma-3-4b-it": "unsloth/gemma-3-4b-it",
    "google/gemma-3-1b-it": "unsloth/gemma-3-1b-it",
}


def _probe_hub_config_download(
    repo_id: str,
    *,
    token: str | None,
) -> Literal["ok", "gated"]:
    if Path(repo_id).exists():
        return "ok"
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id, "config.json", token=token)
        return "ok"
    except Exception as exc:
        if is_gated_hub_error(exc):
            return "gated"
        raise ValueError(
            format_hub_error(exc, context="download", repo_id=repo_id)
        ) from exc


def resolve_trainable_hub_id(
    model_id: str,
    *,
    token: str | None = None,
) -> tuple[str, str | None]:
    """
    Return a Hub repo id that can actually download weights for training.

    When the requested repo is gated (license not accepted), fall back to a known
    trainable mirror and return a user-facing note. Transient Hub/network errors
    are raised instead of silently switching mirrors.
    """
    repo = model_id.strip()
    if not repo or Path(repo).exists():
        return repo, None

    if _probe_hub_config_download(repo, token=token) == "ok":
        return repo, None

    mirror = TRAINABLE_HUB_MIRRORS.get(repo.lower())
    if (
        mirror
        and mirror.lower() != repo.lower()
        and _probe_hub_config_download(mirror, token=token) == "ok"
    ):
        note = (
            f"{repo} is gated on Hugging Face (accept the license at "
            f"https://huggingface.co/{repo} to use the official weights). "
            f"Training with mirror {mirror} instead."
        )
        logger.warning(note)
        return mirror, note

    return repo, (
        f"Cannot download {repo}. If it is gated, open "
        f"https://huggingface.co/{repo} and accept the license, then retry."
    )
