"""Draft / speculative decoding model resolution for inference chat."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from forge.config import ForgeSettings
from forge.db.store import Database
from forge.services.models import resolve_model_path
from seiso.inference.backends import BACKEND_LLAMACPP, BACKEND_TORCH


def _vocab_size_from_path(model_path: str) -> int | None:
    root = Path(model_path)
    if root.is_file():
        root = root.parent
    config = root / "config.json"
    if not config.is_file():
        return None
    try:
        from seiso.io.jsonl import read_json_file

        data = read_json_file(config, default=None)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    for key in ("vocab_size", "padded_vocab_size"):
        value = data.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _assert_draft_compatible(target_path: str | None, draft_path: str) -> None:
    from seiso.inference.backends import (
        gguf_architecture,
        is_dflash_draft,
    )

    if is_dflash_draft(draft_path):
        draft_arch = (gguf_architecture(draft_path) or "").lower()
        if target_path:
            target_arch = (gguf_architecture(target_path) or "").lower()
            if target_arch and draft_arch and "dflash" not in draft_arch:
                target_family = target_arch.split("-", 1)[0]
                draft_family = draft_arch.split("-", 1)[0]
                if target_family and draft_family and target_family != draft_family:
                    raise HTTPException(
                        400,
                        f"Draft architecture {draft_arch!r} is incompatible with target {target_arch!r}",
                    )
        return

    if not target_path:
        return

    target_vocab = _vocab_size_from_path(target_path)
    draft_vocab = _vocab_size_from_path(draft_path)
    if target_vocab is not None and draft_vocab is not None and target_vocab != draft_vocab:
        raise HTTPException(
            400,
            f"Draft/target tokenizers appear incompatible: vocab_size target={target_vocab} draft={draft_vocab}",
        )


async def resolve_draft_model(
    db: Database,
    user_id: str,
    settings: ForgeSettings,
    *,
    draft_model_id: str | None,
    draft_model_path: str | None,
    target_model_path: str | None = None,
) -> dict[str, Any]:
    if draft_model_id and draft_model_path:
        raise HTTPException(403, "Provide draft_model_id or draft_model_path, not both")

    if draft_model_path:
        draft_path = await resolve_model_path(
            db,
            user_id,
            model_id=None,
            model_path=draft_model_path,
            data_dir=settings.data_dir,
        )
        compatibility_checked = False
    elif draft_model_id:
        from forge.services import inference_chat as _inference_chat

        draft_selected = await _inference_chat.get_inference_option(
            db, user_id, draft_model_id
        )
        if not draft_selected:
            raise HTTPException(404, "Draft model not found")
        if not draft_selected.get("selectable", True):
            raise HTTPException(
                400,
                draft_selected.get("hardware_note")
                or "Draft model download is incomplete",
            )
        selected_path = str(draft_selected.get("path") or "")
        if not selected_path:
            raise HTTPException(
                400, "Draft model must be a local safetensors/checkpoint path"
            )
        if target_model_path:
            _assert_draft_compatible(target_model_path, selected_path)
        compatibility_checked = bool(target_model_path)
        draft_path = await resolve_model_path(
            db,
            user_id,
            model_id=draft_model_id,
            model_path=None,
            data_dir=settings.data_dir,
        )
        if not draft_path:
            raise HTTPException(
                400, "Draft model must be a local safetensors/checkpoint path"
            )
    else:
        return {}

    if not draft_path:
        raise HTTPException(400, "Invalid draft model path")

    from seiso.inference.backends import (
        _is_gguf_model,
        _native_linux_requires_isolated_gguf,
        is_dflash_draft,
    )

    if not compatibility_checked:
        _assert_draft_compatible(target_model_path, draft_path)

    is_dflash = is_dflash_draft(draft_path)
    if _is_gguf_model(draft_path, None) and not is_dflash:
        raise HTTPException(
            400,
            "GGUF draft models are only supported for dFlash speculative decoding.",
        )
    if is_dflash and _native_linux_requires_isolated_gguf():
        raise HTTPException(
            400,
            "dFlash speculative decoding uses an in-process llama.cpp GGUF draft, "
            "which is blocked on native Linux NVIDIA. Disable speculative decoding "
            "or set SEISO_LLAMA_ALLOW_INPROCESS_NATIVE_LINUX=1 to explicitly accept "
            "the in-process llama.cpp risk.",
        )

    draft_backend = BACKEND_LLAMACPP if is_dflash else BACKEND_TORCH
    from forge.services import inference_chat as _inference_chat

    _inference_chat.assert_backend_runtime_available(draft_backend)
    _inference_chat.assert_model_fits_for_load(draft_path, mode="chat", backend=draft_backend)
    return {
        "draft_model_path": draft_path,
        "inference_backend": draft_backend,
    }
