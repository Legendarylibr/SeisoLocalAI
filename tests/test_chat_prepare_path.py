"""Shared local-chat prepare/resolve path contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from seiso.inference.backends import BACKEND_LLAMACPP


class _FakeModelDb:
    def __init__(self, row):
        self._row = row

    async def get_model(self, model_id, user_id):
        if self._row and self._row.get("id") == model_id:
            return self._row
        return None

    async def get_model_by_name(self, user_id, model_id):
        if self._row and self._row.get("name") == model_id:
            return self._row
        return None


@pytest.mark.asyncio
async def test_prepare_local_chat_target_missing_model_is_404(monkeypatch):
    from forge.services import inference_chat

    async def missing(*_a, **_k):
        return None

    monkeypatch.setattr(inference_chat, "get_inference_option", missing)
    settings = SimpleNamespace(data_dir="/tmp", model_router_enabled=False)

    with pytest.raises(HTTPException) as exc:
        await inference_chat.prepare_local_chat_target(object(), "u1", settings, model_id="missing")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_prepare_local_chat_target_rejects_incomplete(monkeypatch, tmp_path):
    from forge.services import inference_chat

    async def incomplete(*_a, **_k):
        return {
            "id": "m1",
            "name": "Broken",
            "path": str(tmp_path / "partial.gguf"),
            "format": "gguf",
            "selectable": False,
            "status": "incomplete",
            "hardware_note": "Download incomplete — re-download from Hub.",
        }

    monkeypatch.setattr(inference_chat, "get_inference_option", incomplete)
    settings = SimpleNamespace(data_dir=tmp_path)

    with pytest.raises(HTTPException) as exc:
        await inference_chat.prepare_local_chat_target(object(), "u1", settings, model_id="m1")
    assert exc.value.status_code == 400
    assert "incomplete" in str(exc.value.detail).lower()


@pytest.mark.asyncio
async def test_prepare_sanitizes_n_ctx_and_max_tokens(monkeypatch, tmp_path):
    from forge.services import inference_chat

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")

    async def option(*_a, **_k):
        return {
            "id": "m1",
            "name": "Model",
            "path": str(model_path),
            "format": "gguf",
            "selectable": True,
            "default_backend": BACKEND_LLAMACPP,
            "backends": [BACKEND_LLAMACPP],
            "size_bytes": 10,
        }

    monkeypatch.setattr(inference_chat, "get_inference_option", option)
    monkeypatch.setattr(inference_chat, "assert_model_fits_for_load", lambda *_a, **_k: None)
    monkeypatch.setattr(
        inference_chat,
        "assert_backend_runtime_available",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "seiso.memory.protection.sanitize_inference_payload",
        lambda payload, isolated=False: {
            **payload,
            "max_tokens": 1024,
            "n_ctx": 4096,
        },
    )

    target = await inference_chat.prepare_local_chat_target(
        object(),
        "u1",
        SimpleNamespace(data_dir=tmp_path),
        model_id="m1",
        inference_backend="auto",
        max_tokens=8192,
        n_ctx=65536,
        messages=[{"role": "user", "content": "hi"}],
        sanitize=True,
    )
    assert target["model_path"] == str(model_path)
    assert target["inference_backend"] == BACKEND_LLAMACPP
    assert target["max_tokens"] == 1024
    assert target["n_ctx"] == 4096


@pytest.mark.asyncio
async def test_resolve_preload_uses_shared_prepare(monkeypatch, tmp_path):
    from forge.services import inference_chat

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")

    async def option(*_a, **_k):
        return {
            "id": "m1",
            "name": "Model",
            "path": str(model_path),
            "format": "gguf",
            "selectable": True,
            "default_backend": BACKEND_LLAMACPP,
            "backends": [BACKEND_LLAMACPP],
            "size_bytes": 123,
        }

    monkeypatch.setattr(inference_chat, "get_inference_option", option)
    monkeypatch.setattr(inference_chat, "assert_model_fits_for_load", lambda *_a, **_k: None)
    monkeypatch.setattr(
        inference_chat,
        "assert_backend_runtime_available",
        lambda *_a, **_k: None,
    )

    ctx = await inference_chat.resolve_preload_context(
        object(),
        "u1",
        SimpleNamespace(data_dir=tmp_path),
        "m1",
        "auto",
        max_tokens=2048,
        n_ctx=8192,
    )
    assert ctx["payload"]["n_ctx"] == 8192
    assert ctx["payload"]["max_tokens"] == 2048
    assert ctx["backend"] == BACKEND_LLAMACPP


@pytest.mark.asyncio
async def test_resolve_draft_rejects_vocab_mismatch(monkeypatch, tmp_path):
    from forge.services import inference_chat

    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    (target / "config.json").write_text('{"vocab_size": 32000}', encoding="utf-8")
    (draft / "config.json").write_text('{"vocab_size": 16000}', encoding="utf-8")

    async def draft_option(*_a, **_k):
        return {
            "id": "d1",
            "path": str(draft),
            "format": "safetensors",
            "selectable": True,
        }

    monkeypatch.setattr(inference_chat, "get_inference_option", draft_option)
    monkeypatch.setattr(inference_chat, "assert_model_fits_for_load", lambda *_a, **_k: None)
    monkeypatch.setattr("seiso.inference.backends.is_dflash_draft", lambda _p: False)

    with pytest.raises(HTTPException) as exc:
        await inference_chat.resolve_draft_model(
            _FakeModelDb(
                {
                    "id": "d1",
                    "name": "Draft",
                    "path": str(draft),
                }
            ),
            "u1",
            SimpleNamespace(data_dir=tmp_path),
            draft_model_id="d1",
            draft_model_path=None,
            target_model_path=str(target),
        )
    assert exc.value.status_code == 400
    assert "vocab_size" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_resolve_draft_model_id_validates_inventory_path(monkeypatch, tmp_path):
    from forge.services import inference_chat

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"gguf")

    async def draft_option(*_a, **_k):
        return {
            "id": "d1",
            "path": str(outside),
            "format": "gguf",
            "selectable": True,
        }

    monkeypatch.setattr(inference_chat, "get_inference_option", draft_option)

    with pytest.raises(HTTPException) as exc:
        await inference_chat.resolve_draft_model(
            _FakeModelDb(
                {
                    "id": "d1",
                    "name": "Draft",
                    "path": str(outside),
                }
            ),
            "u1",
            SimpleNamespace(data_dir=sandbox),
            draft_model_id="d1",
            draft_model_path=None,
        )

    assert exc.value.status_code == 403


def test_build_local_option_surfaces_incomplete(monkeypatch, tmp_path):
    from forge.services import inference_models as im

    monkeypatch.setattr(im, "_inventory_artifact_is_complete", lambda *_a, **_k: False)
    monkeypatch.setattr(im, "_installed_backends", lambda: {BACKEND_LLAMACPP: True})

    row = {
        "id": "m1",
        "name": "Partial",
        "path": str(tmp_path / "p.gguf"),
        "format": "gguf",
        "source": "hf:org/model",
        "size_bytes": 1,
        "metadata_json": "{}",
    }
    opt = im._build_local_option(row, installed={BACKEND_LLAMACPP: True}, profile=None)
    assert opt is not None
    assert opt["selectable"] is False
    assert opt["status"] == "incomplete"


def test_resolve_chat_target_missing_option_raises():
    from forge.services.inference_models import resolve_chat_target

    with pytest.raises(ValueError, match="not found"):
        resolve_chat_target(None, model_id="stale", inference_backend="auto")


def test_dead_compat_helpers_removed():
    import forge.api.routes.compat as compat_routes

    assert not hasattr(compat_routes, "_resolve_openai_model_path")
    assert not hasattr(compat_routes, "_resolve_payload")
    assert not hasattr(compat_routes, "_default_openai_gguf_backend")
    assert hasattr(compat_routes, "_prepare_compat_chat_payload")
