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
from seiso.security import USER_SCOPED_DATA_ROOTS, SecurityError


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
        "nostr_keys",
    )
    for name in expected_subdirs:
        assert (settings.data_dir / name).is_dir(), name

    assert resolve_hf_cache_dir(settings.data_dir) == settings.data_dir / "hf_cache"

    uid = "user-1"
    model_dir = user_dir(settings.data_dir, uid, "models")
    assert model_dir == settings.data_dir / "models" / uid
    model_dir.mkdir(parents=True, exist_ok=True)
    assert model_dir.is_dir()


def test_user_dir_uses_shared_scoped_roots(tmp_path: Path):
    """Forge must not maintain a divergent root set (S1-003)."""
    for category in USER_SCOPED_DATA_ROOTS:
        path = user_dir(tmp_path, "user-1", category)
        assert path == tmp_path / category / "user-1"
    with pytest.raises(SecurityError, match="Unknown user path category"):
        user_dir(tmp_path, "user-1", "hf_cache")


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
    (install_root / "data" / "sample.jsonl").write_text(
        '{"messages":[]}\n', encoding="utf-8"
    )
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


def test_resolve_training_dataset_path_finds_checkpoints_relative(tmp_path: Path):
    """S1-004: relative paths resolve under any USER_SCOPED_DATA_ROOTS category."""
    user_id = "user-1"
    ds = tmp_path / "checkpoints" / user_id / "train.jsonl"
    ds.parent.mkdir(parents=True)
    ds.write_text('{"text":"x"}\n', encoding="utf-8")
    resolved = resolve_training_dataset_path(tmp_path, user_id, "train.jsonl")
    assert Path(resolved) == ds.resolve()


def test_is_local_filesystem_path_matches_dataset_helper():
    from forge.services.user_paths import is_local_filesystem_path
    from seiso.training.datasets import looks_like_local_dataset_path

    cases = [
        "/abs/path.jsonl",
        "uploads/u/x.jsonl",
        "org/hub-dataset",
        "./relative.jsonl",
        "plain-name",
    ]
    for case in cases:
        assert is_local_filesystem_path(case) == looks_like_local_dataset_path(case)


def test_assert_llama_cpp_binary_allows_venv_path(tmp_path: Path):
    binary = tmp_path / ".venv" / "bin" / "llama-cli"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"fake-binary")

    resolved = assert_llama_cpp_binary(binary)
    assert resolved == binary.resolve()


def test_assert_llama_cpp_binary_rejects_tmp():
    import os

    from seiso.security import SecurityError

    binary = Path("/tmp") / f"seiso_llama_test_{os.getpid()}"
    binary.write_bytes(b"fake-binary")
    try:
        with pytest.raises(SecurityError, match="temporary"):
            assert_llama_cpp_binary(binary)
    finally:
        binary.unlink(missing_ok=True)
