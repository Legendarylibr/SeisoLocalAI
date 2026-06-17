"""Tests for model download orchestration guards."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from forge.db.crypto import generate_encryption_key
from forge.db.store import Database
from forge.services import model_download


def test_disk_space_guard_allows_unknown_size(tmp_path):
    model_download._assert_disk_space_for_download(tmp_path, 0)


def test_disk_space_guard_raises_when_cache_disk_too_small(monkeypatch, tmp_path):
    monkeypatch.setattr(
        model_download.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1_000_000_000),
    )

    with pytest.raises(ValueError, match="Need about 4.9 GB"):
        model_download._assert_disk_space_for_download(tmp_path, 5_000_000_000)


@pytest.mark.asyncio
async def test_perform_gguf_download_registers_llamacpp_inventory(monkeypatch, tmp_path):
    cached = tmp_path / "hf_cache" / "model-q4.gguf"
    cached.parent.mkdir()
    cached.write_bytes(b"gguf-bytes")

    monkeypatch.setattr(model_download, "assert_hub_ready_for_download", lambda **_kwargs: None)
    monkeypatch.setattr(model_download, "resolve_hf_token", lambda **_kwargs: (None, "none"))
    monkeypatch.setattr(model_download, "get_by_repo", lambda _repo: None)
    monkeypatch.setattr(
        model_download,
        "resolve_gguf_artifact",
        lambda *_args, **_kwargs: {
            "gguf_repo": "mirror/Model-GGUF",
            "filename": "model-q4.gguf",
            "size_bytes": cached.stat().st_size,
        },
    )
    monkeypatch.setattr(model_download, "_assert_disk_space_for_download", lambda *_args: None)
    monkeypatch.setattr(
        model_download,
        "download_gguf",
        lambda *_args, **_kwargs: {
            "path": str(cached),
            "filename": "model-q4.gguf",
            "inventory_name": "org--Model/model-q4.gguf",
        },
    )

    db = Database(tmp_path / "forge.db", encryption_key=generate_encryption_key(), ephemeral=True)
    result = await model_download.perform_model_download(
        user_id="u1",
        db=db,
        data_dir=tmp_path,
        hf_cache_dir=tmp_path / "hf_cache",
        settings_hf_token=None,
        db_encryption_key=generate_encryption_key(),
        repo_id="org/Model",
        variant="gguf",
    )

    rows = await db.list_models("u1")
    assert result["variant"] == "gguf"
    assert result["gguf_repo"] == "mirror/Model-GGUF"
    assert rows[0]["format"] == "gguf"
    assert rows[0]["source"] == "hf:org/Model"
    assert rows[0]["size_bytes"] == cached.stat().st_size
    assert (tmp_path / "models" / "u1" / "org--Model" / "model-q4.gguf").is_symlink()
