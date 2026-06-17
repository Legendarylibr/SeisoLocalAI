"""Tests for Hugging Face model cards and publish validation."""

from pathlib import Path

from forge.services import hf_auth
from forge.services.hf_auth import (
    find_hf_cli,
    hf_auth_status,
    resolve_hf_token,
    resolve_hf_token_for_download,
    save_user_hf_token,
)
from forge.services.publishable import is_pushable_model
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
    assert is_pushable_model({"source": "rl_quant"})
    assert not is_pushable_model({"source": "hf:meta-llama/Llama"})
    assert not is_pushable_model({"source": "scan"})
    assert not is_pushable_model({"source": "manual"})


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


def test_resolve_hf_token_for_download_drops_invalid_token_when_anonymous_down(monkeypatch):
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
