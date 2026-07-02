"""Tests for local model inventory registration helpers."""

from __future__ import annotations

import pytest

from forge.db.crypto import generate_encryption_key
from forge.db.store import Database
from forge.services import model_registry


@pytest.mark.asyncio
async def test_register_model_path_skips_size_scan_for_existing_path(monkeypatch, tmp_path):
    model_dir = tmp_path / "models" / "existing"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"weights")

    db = Database(tmp_path / "forge.db", encryption_key=generate_encryption_key(), ephemeral=True)
    existing = await db.add_model(
        user_id="u1",
        name="existing",
        path=str(model_dir),
        source="export",
        format="safetensors",
        size_bytes=7,
    )

    class FastDuplicateDb:
        async def get_model_by_path(self, user_id: str, path: str) -> dict | None:
            assert user_id == "u1"
            assert path == str(model_dir)
            return existing

        async def list_models(self, _user_id: str) -> list[dict]:
            pytest.fail("exact duplicate path should not scan inventory")

    monkeypatch.setattr(
        model_registry,
        "model_weight_size_bytes",
        lambda _path: pytest.fail("duplicate registration should not scan size"),
    )

    result = await model_registry.register_model_path(
        FastDuplicateDb(),
        user_id="u1",
        data_dir=tmp_path,
        path=model_dir,
        name="existing",
        source="export",
        model_format="safetensors",
    )

    assert result is None
