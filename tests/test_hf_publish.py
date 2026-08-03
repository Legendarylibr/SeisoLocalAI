"""Tests for Hugging Face model cards and publish validation."""

from pathlib import Path

import pytest

from forge.config import ForgeSettings
from forge.services import hf_auth
from forge.services.hf_auth import (
    find_hf_cli,
    hf_auth_status,
    resolve_hf_token,
    resolve_hf_token_for_download,
    resolve_hf_token_for_upload,
    save_user_hf_token,
)
from forge.services.hub_publish import HubPublishRequest, resolve_hub_publish_token
from forge.services.publishable import (
    assert_pushable_checkpoint,
    assert_pushable_path,
    get_model_for_user,
    is_pushable_model,
)
from seiso.export.model_card import HubModelMetadata, render_readme, write_hub_artifacts


def test_hub_metadata_repo_id():
    meta = HubModelMetadata(username="alice", model_name="my-model", author="Alice")
    assert meta.repo_id == "alice/my-model"


def test_model_card_frontmatter():
    meta = HubModelMetadata(
        username="alice",
        model_name="my-model",
        author="Alice",
        base_model="meta-llama/Llama-3.2-3B",
        quantizations=["q4_k_m"],
    )
    readme = render_readme(meta)
    assert "---" in readme
    assert "base_model: meta-llama/Llama-3.2-3B" in readme
    assert "Alice" in readme
    assert "q4_k_m" in readme


def test_write_hub_artifacts(tmp_path: Path):
    meta = HubModelMetadata(username="bob", model_name="export", author="Bob")
    paths = write_hub_artifacts(tmp_path, meta)
    assert paths["readme"].is_file()
    assert paths["metadata"].is_file()
    assert "bob/export" in paths["metadata"].read_text()


def test_pushable_sources():
    assert is_pushable_model({"source": "export"})
    assert is_pushable_model({"source": "training"})
    assert is_pushable_model({"source": "export:job-1:merged"})
    assert is_pushable_model({"source": "training:job-1"})
    assert not is_pushable_model({"source": "hf:meta-llama/Llama"})
    assert not is_pushable_model({"source": "scan"})
    assert not is_pushable_model({"source": "manual"})


@pytest.mark.asyncio
async def test_get_model_for_user_uses_single_model_lookup():
    class FakeDb:
        async def get_model(self, model_id: str, user_id: str) -> dict:
            return {"id": model_id, "user_id": user_id}

        async def list_models(self, _user_id: str) -> list[dict]:
            pytest.fail("single model lookup should not scan the full inventory")

    assert await get_model_for_user(FakeDb(), "m1", "u1") == {
        "id": "m1",
        "user_id": "u1",
    }


@pytest.mark.asyncio
async def test_assert_pushable_path_allows_exports_without_inventory_scan(tmp_path):
    export_path = tmp_path / "exports" / "u1" / "job" / "model.gguf"
    export_path.parent.mkdir(parents=True)
    export_path.write_bytes(b"gguf")

    class FakeDb:
        async def list_models(self, _user_id: str) -> list[dict]:
            pytest.fail("exports path should not scan model inventory")

    assert (
        await assert_pushable_path(
            FakeDb(),
            data_dir=tmp_path,
            user_id="u1",
            target=export_path,
        )
        == export_path
    )


@pytest.mark.asyncio
async def test_assert_pushable_path_uses_exact_path_lookup(tmp_path):
    model_path = tmp_path / "models" / "u1" / "export" / "model.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"gguf")

    class FakeDb:
        async def get_model_by_path(self, user_id: str, path: str) -> dict | None:
            assert user_id == "u1"
            if path == str(model_path):
                return {"source": "export", "path": path}
            return None

        async def list_models(self, _user_id: str) -> list[dict]:
            pytest.fail("exact registered path should not scan model inventory")

    assert (
        await assert_pushable_path(
            FakeDb(),
            data_dir=tmp_path,
            user_id="u1",
            target=model_path,
        )
        == model_path
    )


@pytest.mark.asyncio
async def test_assert_pushable_checkpoint_uses_exact_path_lookup(tmp_path):
    model_path = tmp_path / "models" / "u1" / "merged"
    model_path.mkdir(parents=True)
    (model_path / "model.safetensors").write_bytes(b"weights")

    class FakeDb:
        async def get_model_by_path(self, user_id: str, path: str) -> dict | None:
            assert user_id == "u1"
            if path == str(model_path):
                return {"source": "training", "path": path}
            return None

        async def list_models(self, _user_id: str) -> list[dict]:
            pytest.fail("exact checkpoint path should not scan model inventory")

    assert (
        await assert_pushable_checkpoint(
            FakeDb(),
            data_dir=tmp_path,
            user_id="u1",
            checkpoint=model_path,
        )
        == model_path
    )


def test_resolve_hf_token_priority(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    token, source = resolve_hf_token(request_token="hf_req", settings_token="hf_env")
    assert token == "hf_req"
    assert source == "request"

    token, source = resolve_hf_token(settings_token="hf_env")
    assert token == "hf_env"
    assert source == "env_seiso"


def test_user_hf_token_store(tmp_path):
    from forge.db.crypto import generate_encryption_key

    key = generate_encryption_key()
    save_user_hf_token(tmp_path, "user1", "hf_secret", encryption_key=key)
    token, source = resolve_hf_token(
        user_id="user1",
        data_dir=tmp_path,
        encryption_key=key,
    )
    assert token == "hf_secret"
    assert source == "user_store"


def test_resolve_hf_token_for_download_drops_invalid_token(monkeypatch):
    monkeypatch.setattr(
        "forge.services.hf_connectivity.probe_hf_hub",
        lambda **_: type(
            "R",
            (),
            {"token_valid": False, "token_invalid": True, "anonymous_ok": True},
        )(),
    )
    token, source = resolve_hf_token_for_download(settings_token="hf_bad")
    assert token is None
    assert source == "none"


def test_resolve_hf_token_for_upload_requires_valid_token(monkeypatch):
    monkeypatch.setattr(
        "forge.services.hf_connectivity.probe_hf_hub",
        lambda **_: type(
            "R",
            (),
            {"token_valid": True, "token_invalid": False, "anonymous_ok": True},
        )(),
    )
    token, source = resolve_hf_token_for_upload(settings_token="hf_good")
    assert token == "hf_good"
    assert source == "env_seiso"


def test_resolve_hf_token_for_upload_rejects_invalid_token(monkeypatch):
    monkeypatch.setattr(
        "forge.services.hf_connectivity.probe_hf_hub",
        lambda **_: type(
            "R",
            (),
            {"token_valid": False, "token_invalid": True, "anonymous_ok": True},
        )(),
    )
    token, source = resolve_hf_token_for_upload(settings_token="hf_bad")
    assert token is None
    assert source == "none"


def test_resolve_hf_token_for_download_drops_invalid_token_when_anonymous_down(
    monkeypatch,
):
    monkeypatch.setattr(
        "forge.services.hf_connectivity.probe_hf_hub",
        lambda **_: type(
            "R",
            (),
            {"token_valid": False, "token_invalid": True, "anonymous_ok": False},
        )(),
    )
    token, source = resolve_hf_token_for_download(settings_token="hf_bad")
    assert token is None
    assert source == "none"


def test_hf_auth_status_no_token():
    status = hf_auth_status()
    assert status.token_configured is False or status.token_sources


def test_resolve_hub_publish_token_requires_valid_token(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "forge.services.hf_connectivity.probe_hf_hub",
        lambda **_: type(
            "R",
            (),
            {"token_valid": False, "token_invalid": True, "anonymous_ok": True},
        )(),
    )
    settings = ForgeSettings(data_dir=tmp_path, hf_token="hf_bad")
    hub = HubPublishRequest(
        username="alice",
        model_name="my-model",
        author="Alice",
        hf_token="hf_bad",
    )
    assert resolve_hub_publish_token(settings, "user1", hub) is None


def test_resolve_hub_publish_token_accepts_valid_token(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "forge.services.hf_connectivity.probe_hf_hub",
        lambda **_: type(
            "R",
            (),
            {"token_valid": True, "token_invalid": False, "anonymous_ok": True},
        )(),
    )
    settings = ForgeSettings(data_dir=tmp_path, hf_token="hf_good")
    hub = HubPublishRequest(
        username="alice",
        model_name="my-model",
        author="Alice",
    )
    assert resolve_hub_publish_token(settings, "user1", hub) == "hf_good"


def test_find_hf_cli_checks_active_python_bin(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    target_python = tmp_path / "python-real"
    target_python.write_text("", encoding="utf-8")
    python = bin_dir / "python"
    python.symlink_to(target_python)
    hf = bin_dir / "hf"
    hf.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(hf_auth.sys, "executable", str(python))
    monkeypatch.setenv("HF_CLI", "")
    monkeypatch.setattr(hf_auth.shutil, "which", lambda _name: None)

    assert find_hf_cli() == str(hf)
