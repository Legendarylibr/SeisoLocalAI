"""Tests for model download orchestration guards."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from forge.db.crypto import generate_encryption_key
from forge.db.store import Database
from forge.services import model_download
from forge.services.hf_cache_inventory import sync_hf_cache_inventory


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


def test_disk_space_guard_checks_xet_cache(monkeypatch, tmp_path):
    cache = tmp_path / "hf_cache"
    xet_cache = tmp_path / "hf_xet_cache"
    monkeypatch.setenv("HF_XET_CACHE", str(xet_cache))

    def fake_disk_usage(path):
        free = 10_000_000_000
        if path == xet_cache.resolve():
            free = 1_000_000_000
        return SimpleNamespace(free=free)

    monkeypatch.setattr(model_download.shutil, "disk_usage", fake_disk_usage)

    with pytest.raises(ValueError, match="hf_xet_cache"):
        model_download._assert_disk_space_for_download(cache, 5_000_000_000)


def test_cached_download_rejects_incomplete_gguf(tmp_path):
    cached = tmp_path / "model-Q4_K_M.gguf"
    cached.write_bytes(b"partial")

    result = model_download._cached_download_result_if_usable(
        {
            "id": "m1",
            "path": str(cached),
            "format": "gguf",
            "size_bytes": 10_000,
            "metadata_json": "{}",
        },
        repo_id="org/Model",
        variant="gguf",
    )

    assert result is None


def test_cached_download_rejects_stale_partial_gguf_inventory(monkeypatch, tmp_path):
    cached = tmp_path / "model-Q4_K_M.gguf"
    cached.write_bytes(b"partial")

    monkeypatch.setattr(
        model_download,
        "get_gguf_file_size_bytes",
        lambda _repo, _filename: 10_000,
    )

    result = model_download._cached_download_result_if_usable(
        {
            "id": "m1",
            "path": str(cached),
            "format": "gguf",
            "size_bytes": cached.stat().st_size,
            "metadata_json": (
                '{"repo_id": "org/Model", "gguf_repo": "mirror/Model-GGUF", '
                '"gguf_files": ["model-Q4_K_M.gguf"]}'
            ),
        },
        repo_id="org/Model",
        variant="gguf",
    )

    assert result is None


def test_cached_download_rejects_hf_gguf_without_metadata(tmp_path):
    cached = tmp_path / "model-Q4_K_M.gguf"
    cached.write_bytes(b"partial")

    result = model_download._cached_download_result_if_usable(
        {
            "id": "m1",
            "path": str(cached),
            "source": "hf:org/Model",
            "format": "gguf",
            "size_bytes": cached.stat().st_size,
            "metadata_json": "{}",
        },
        repo_id="org/Model",
        variant="gguf",
    )

    assert result is None


def test_cached_download_validates_specific_gguf_files_in_directory(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    q4 = model_dir / "model-Q4_K_M.gguf"
    q8 = model_dir / "model-Q8_0.gguf"
    q4.write_bytes(b"x" * 10)
    q8.write_bytes(b"y" * 100)

    monkeypatch.setattr(
        model_download,
        "get_gguf_file_size_bytes",
        lambda _repo, filename: 50 if filename.endswith("Q4_K_M.gguf") else 100,
    )

    result = model_download._cached_download_result_if_usable(
        {
            "id": "m1",
            "path": str(model_dir),
            "source": "hf:org/Model",
            "format": "gguf",
            "size_bytes": q4.stat().st_size + q8.stat().st_size,
            "metadata_json": (
                '{"repo_id": "org/Model", "gguf_repo": "mirror/Model-GGUF", '
                '"gguf_files": ["model-Q4_K_M.gguf"]}'
            ),
        },
        repo_id="org/Model",
        variant="gguf",
    )

    assert result is None


@pytest.mark.asyncio
async def test_perform_gguf_download_registers_llamacpp_inventory(monkeypatch, tmp_path):
    cached = tmp_path / "hf_cache" / "model-q4.gguf"
    cached.parent.mkdir()
    cached.write_bytes(b"gguf-bytes")

    monkeypatch.setattr(model_download, "assert_hub_ready_for_download", lambda **_kwargs: None)
    monkeypatch.setattr(model_download, "resolve_hf_token_for_download", lambda **_kwargs: (None, "none"))
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


@pytest.mark.asyncio
async def test_perform_download_returns_cached_inventory_without_redownload(monkeypatch, tmp_path):
    cached = tmp_path / "hf_cache" / "model-q4.gguf"
    cached.parent.mkdir()
    cached.write_bytes(b"gguf-bytes")
    inv = tmp_path / "models" / "u1" / "org--Model" / "model-q4.gguf"
    inv.parent.mkdir(parents=True)
    inv.symlink_to(cached)

    db = Database(tmp_path / "forge.db", encryption_key=generate_encryption_key(), ephemeral=True)
    row = await db.add_model(
        user_id="u1",
        source="hf:org/Model",
        name="model-q4.gguf",
        path=str(inv),
        format="gguf",
        size_bytes=cached.stat().st_size,
        metadata={
            "repo_id": "org/Model",
            "cache_dir": str(tmp_path / "hf_cache"),
            "gguf_file": "model-q4.gguf",
            "gguf_files": ["model-q4.gguf"],
        },
    )
    monkeypatch.setattr(
        model_download,
        "get_gguf_file_size_bytes",
        lambda _repo, _filename: cached.stat().st_size,
    )

    monkeypatch.setattr(
        model_download,
        "_sync_download_artifacts",
        lambda **_kwargs: pytest.fail("download should not run"),
    )

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

    assert result["cached"] is True
    assert result["model_id"] == row["id"]


def test_link_inventory_preserves_hf_snapshot_symlink(tmp_path):
    blob = tmp_path / "hf_cache" / "models--org--Model-GGUF" / "blobs" / "abc"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"gguf-bytes")
    snapshot_file = (
        tmp_path
        / "hf_cache"
        / "models--org--Model-GGUF"
        / "snapshots"
        / "rev"
        / "model-q4.gguf"
    )
    snapshot_file.parent.mkdir(parents=True)
    snapshot_file.symlink_to("../../blobs/abc")

    inv = model_download.link_inventory(
        tmp_path / "models" / "u1",
        "org--Model/model-q4.gguf",
        snapshot_file,
    )

    assert inv.is_symlink()
    assert inv.readlink() == snapshot_file.absolute()


@pytest.mark.asyncio
async def test_find_inventory_for_catalog_repo_matches_metadata(monkeypatch, tmp_path):
    cached = tmp_path / "hf_cache" / "model-q4.gguf"
    cached.parent.mkdir()
    cached.write_bytes(b"gguf-bytes")
    inv = tmp_path / "models" / "u1" / "mirror--Model-GGUF" / "model-q4.gguf"
    inv.parent.mkdir(parents=True)
    inv.symlink_to(cached)

    db = Database(tmp_path / "forge.db", encryption_key=generate_encryption_key(), ephemeral=True)
    await db.add_model(
        user_id="u1",
        source="hf:mirror/Model-GGUF",
        name="model-q4.gguf",
        path=str(inv),
        format="gguf",
        size_bytes=cached.stat().st_size,
        metadata={
            "repo_id": "org/Model",
            "gguf_repo": "mirror/Model-GGUF",
            "gguf_file": "model-q4.gguf",
            "gguf_files": ["model-q4.gguf"],
        },
    )
    monkeypatch.setattr(
        model_download,
        "get_gguf_file_size_bytes",
        lambda _repo, _filename: cached.stat().st_size,
    )

    found = await model_download.find_inventory_for_catalog_repo(db, "u1", "org/Model")
    assert found is not None
    assert found["source"] == "hf:mirror/Model-GGUF"

    monkeypatch.setattr(
        model_download,
        "_sync_download_artifacts",
        lambda **_kwargs: pytest.fail("download should not run"),
    )
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
    assert result["cached"] is True


@pytest.mark.asyncio
async def test_sync_hf_cache_inventory_registers_cached_gguf(tmp_path):
    snapshot = tmp_path / "hf_cache" / "models--org--Model-GGUF" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    gguf = snapshot / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"gguf-bytes")

    db = Database(tmp_path / "forge.db", encryption_key=generate_encryption_key(), ephemeral=True)
    count = await sync_hf_cache_inventory(
        db,
        "u1",
        data_dir=tmp_path,
        hf_cache_dir=tmp_path / "hf_cache",
    )

    rows = await db.list_models("u1")
    assert count == 1
    assert rows[0]["source"] == "hf:org/Model-GGUF"
    assert rows[0]["format"] == "gguf"
    assert (tmp_path / "models" / "u1" / "org--Model-GGUF" / "model-Q4_K_M.gguf").is_symlink()


@pytest.mark.asyncio
async def test_sync_hf_cache_inventory_skips_partial_catalog_gguf(monkeypatch, tmp_path):
    snapshot = tmp_path / "hf_cache" / "models--mirror--Model-GGUF" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    gguf = snapshot / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"partial")

    monkeypatch.setattr(
        "forge.services.hf_cache_inventory._catalog_entry_for_cached_repo",
        lambda _repo: SimpleNamespace(repo_id="org/Model", quant="Q4_K_M"),
    )
    monkeypatch.setattr(
        "forge.services.hf_cache_inventory.get_gguf_file_size_bytes",
        lambda _repo, _filename: 10_000,
    )

    db = Database(tmp_path / "forge.db", encryption_key=generate_encryption_key(), ephemeral=True)
    count = await sync_hf_cache_inventory(
        db,
        "u1",
        data_dir=tmp_path,
        hf_cache_dir=tmp_path / "hf_cache",
    )

    assert count == 0
    assert await db.list_models("u1") == []


@pytest.mark.asyncio
async def test_sync_hf_cache_inventory_skips_partial_safetensors(monkeypatch, tmp_path):
    snapshot = tmp_path / "hf_cache" / "models--org--Model" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    weights = snapshot / "model.safetensors"
    weights.write_bytes(b"partial")

    monkeypatch.setattr(
        "forge.services.hf_cache_inventory._catalog_entry_for_cached_repo",
        lambda _repo: SimpleNamespace(repo_id="org/Model"),
    )
    monkeypatch.setattr(
        "forge.services.hf_cache_inventory.estimate_snapshot_download_bytes",
        lambda _repo: 10_000,
    )

    db = Database(tmp_path / "forge.db", encryption_key=generate_encryption_key(), ephemeral=True)
    count = await sync_hf_cache_inventory(
        db,
        "u1",
        data_dir=tmp_path,
        hf_cache_dir=tmp_path / "hf_cache",
    )

    assert count == 0
    assert await db.list_models("u1") == []


def test_resolve_download_variant_prefers_safetensors_without_llamacpp(monkeypatch):
    monkeypatch.setattr(
        model_download,
        "check_inference_runtime",
        lambda: SimpleNamespace(llamacpp=False, mlx=True, torch=False),
    )
    assert model_download.resolve_download_variant("auto") == "safetensors"


def test_resolve_download_variant_prefers_gguf_with_llamacpp(monkeypatch):
    monkeypatch.setattr(
        model_download,
        "check_inference_runtime",
        lambda: SimpleNamespace(llamacpp=True, mlx=True, torch=True),
    )
    assert model_download.resolve_download_variant("auto") == "gguf"


def test_cached_download_rejects_gguf_when_safetensors_requested(tmp_path):
    cached = tmp_path / "model-Q4_K_M.gguf"
    cached.write_bytes(b"x" * 100)

    result = model_download._cached_download_result_if_usable(
        {
            "id": "m1",
            "path": str(cached),
            "format": "gguf",
            "size_bytes": 100,
            "metadata_json": "{}",
        },
        repo_id="org/Model",
        variant="safetensors",
    )

    assert result is None


@pytest.mark.asyncio
async def test_perform_download_auto_skips_gguf_cache_when_safetensors_preferred(monkeypatch, tmp_path):
    monkeypatch.setattr(
        model_download,
        "check_inference_runtime",
        lambda: SimpleNamespace(llamacpp=False, mlx=True, torch=False),
    )
    monkeypatch.setattr(
        model_download,
        "_sync_download_artifacts",
        lambda **_kwargs: {
            "variant": "safetensors",
            "source": "hf:org/Model",
            "name": "Model",
            "path": str(tmp_path / "models" / "u1" / "org--Model"),
            "format": "safetensors",
            "size_bytes": 1000,
            "metadata": {"repo_id": "org/Model"},
            "downloaded": [str(tmp_path / "snap")],
            "repo_id": "org/Model",
            "cache_dir": str(tmp_path / "hf_cache"),
        },
    )

    cached = tmp_path / "hf_cache" / "model-q4.gguf"
    cached.parent.mkdir()
    cached.write_bytes(b"gguf-bytes")
    inv = tmp_path / "models" / "u1" / "org--Model" / "model-q4.gguf"
    inv.parent.mkdir(parents=True)
    inv.symlink_to(cached)

    db = Database(tmp_path / "forge.db", encryption_key=generate_encryption_key(), ephemeral=True)
    await db.add_model(
        user_id="u1",
        source="hf:org/Model",
        name="model-q4.gguf",
        path=str(inv),
        format="gguf",
        size_bytes=cached.stat().st_size,
        metadata={"repo_id": "org/Model"},
    )

    result = await model_download.perform_model_download(
        user_id="u1",
        db=db,
        data_dir=tmp_path,
        hf_cache_dir=tmp_path / "hf_cache",
        settings_hf_token=None,
        db_encryption_key=generate_encryption_key(),
        repo_id="org/Model",
        variant="auto",
    )

    assert result["variant"] == "safetensors"
    assert "cached" not in result


@pytest.mark.asyncio
async def test_sync_hf_cache_inventory_registers_cached_safetensors(tmp_path):
    snapshot = tmp_path / "hf_cache" / "models--org--Model" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    weights = snapshot / "model.safetensors"
    weights.write_bytes(b"weights")

    db = Database(tmp_path / "forge.db", encryption_key=generate_encryption_key(), ephemeral=True)
    count = await sync_hf_cache_inventory(
        db,
        "u1",
        data_dir=tmp_path,
        hf_cache_dir=tmp_path / "hf_cache",
    )

    rows = await db.list_models("u1")
    assert count == 1
    assert rows[0]["source"] == "hf:org/Model"
    assert rows[0]["format"] == "safetensors"
    link = tmp_path / "models" / "u1" / "org--Model"
    assert link.is_symlink()
    assert link.resolve() == snapshot.resolve()
