"""Tests for per-user path policy with inventory symlinks."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.config import ForgeSettings
from forge.services.user_paths import (
    assert_llama_cpp_binary,
    assert_user_path,
    resolve_training_dataset_path,
    user_dir,
)
from seiso.models.hf_env import resolve_hf_cache_dir
from seiso.security import SecurityError


def test_data_dir_layout_matches_docs(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_XET_CACHE", raising=False)
    settings = ForgeSettings()
    settings.ensure_dirs()

    expected_subdirs = (
        "models",
        "checkpoints",
        "exports",
        "knowledge",
        "sandbox",
        "artifacts",
        "recipes",
        "uploads",
        "rl_quant",
        "compress",
        "distill_rl",
        "hf_cache",
        "hf_tokens",
    )
    for name in expected_subdirs:
        assert (settings.data_dir / name).is_dir(), name

    assert resolve_hf_cache_dir(settings.data_dir) == settings.data_dir / "hf_cache"

    uid = "user-1"
    model_dir = user_dir(settings.data_dir, uid, "models")
    assert model_dir == settings.data_dir / "models" / uid
    model_dir.mkdir(parents=True, exist_ok=True)
    assert model_dir.is_dir()


def test_assert_user_path_allows_inventory_symlink(tmp_path: Path):
    user_id = "user-1"
    cache_file = tmp_path / "hf_cache" / "model-q4.gguf"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"gguf")

    inventory = tmp_path / "models" / user_id / "repo" / "model-q4.gguf"
    inventory.parent.mkdir(parents=True)
    inventory.symlink_to(cache_file)

    resolved = assert_user_path(tmp_path, user_id, inventory)
    assert resolved == cache_file.resolve()


def test_assert_user_path_rejects_path_outside_user_tree(tmp_path: Path):
    outsider = tmp_path / "models" / "other-user" / "model.gguf"
    outsider.parent.mkdir(parents=True)
    outsider.write_bytes(b"x")

    with pytest.raises(SecurityError, match="Path must be under"):
        assert_user_path(tmp_path, "user-1", outsider)


def test_assert_user_path_rejects_symlink_escape(tmp_path: Path):
    user_id = "user-1"
    uploads = tmp_path / "uploads" / user_id
    uploads.mkdir(parents=True)
    link = uploads / "evil.txt"
    link.symlink_to("/etc/passwd")

    with pytest.raises(SecurityError, match="outside sandbox"):
        assert_user_path(tmp_path, user_id, link)


def test_assert_user_path_rejects_cross_user_symlink(tmp_path: Path):
    user_id = "user-a"
    victim = tmp_path / "models" / "user-b" / "secret.gguf"
    victim.parent.mkdir(parents=True)
    victim.write_bytes(b"secret")

    link = tmp_path / "models" / user_id / "stolen.gguf"
    link.parent.mkdir(parents=True)
    link.symlink_to(victim)

    with pytest.raises(SecurityError, match="Path must be under"):
        assert_user_path(tmp_path, user_id, link)


def test_resolve_training_dataset_path_seeds_sample_jsonl(tmp_path: Path):
    install_root = tmp_path / "install"
    (install_root / "data").mkdir(parents=True)
    (install_root / "data" / "sample.jsonl").write_text('{"messages":[]}\n', encoding="utf-8")
    user_id = "user-1"

    resolved = resolve_training_dataset_path(
        tmp_path,
        user_id,
        "./data/sample.jsonl",
        install_root=install_root,
    )
    expected = tmp_path / "uploads" / user_id / "sample.jsonl"
    assert Path(resolved) == expected
    assert expected.read_text(encoding="utf-8") == '{"messages":[]}\n'


def test_assert_llama_cpp_binary_allows_venv_path(tmp_path: Path):
    binary = tmp_path / ".venv" / "bin" / "llama-cli"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"fake-binary")

    resolved = assert_llama_cpp_binary(binary)
    assert resolved == binary.resolve()
