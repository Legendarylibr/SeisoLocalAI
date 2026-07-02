"""Tests for local model path resolution helpers."""

from __future__ import annotations

import pytest

from forge.services.models import resolve_model_path


@pytest.mark.asyncio
async def test_resolve_model_path_uses_single_row_lookup_for_ids(tmp_path):
    model_root = tmp_path / "models" / "u1"
    model_root.mkdir(parents=True)
    model_path = model_root / "model.gguf"
    model_path.write_bytes(b"gguf")

    class FakeDb:
        async def get_model(self, model_id: str, user_id: str) -> dict:
            return {
                "id": model_id,
                "user_id": user_id,
                "name": "model",
                "path": str(model_path),
            }

        async def get_model_by_name(self, _user_id: str, _name: str) -> dict | None:
            pytest.fail("id lookup should not query by name")

        async def list_models(self, _user_id: str) -> list[dict]:
            pytest.fail("id lookup should not scan model inventory")

    assert await resolve_model_path(
        FakeDb(),
        "u1",
        model_id="m1",
        model_path=None,
        data_dir=tmp_path,
    ) == str(model_path)
