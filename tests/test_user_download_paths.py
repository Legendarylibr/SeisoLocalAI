"""Tests for sandbox-safe download path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.services.user_paths import pick_user_download_file
from seiso.security import SecurityError


def test_pick_user_download_file_rejects_symlink_escape(tmp_path: Path):
    user_id = "user-1"
    model_dir = tmp_path / "models" / user_id / "bundle"
    model_dir.mkdir(parents=True)
    link = model_dir / "leak.gguf"
    link.symlink_to("/etc/passwd")

    with pytest.raises(SecurityError, match="outside sandbox"):
        pick_user_download_file(tmp_path, user_id, model_dir, pattern="*.gguf")


def test_pick_user_download_file_rejects_absolute_metadata_name(tmp_path: Path):
    user_id = "user-1"
    model_dir = tmp_path / "models" / user_id / "bundle"
    model_dir.mkdir(parents=True)
    real = model_dir / "model.gguf"
    real.write_bytes(b"gguf")

    with pytest.raises(SecurityError):
        pick_user_download_file(
            tmp_path,
            user_id,
            model_dir,
            relative_name="/etc/passwd",
        )


def test_pick_user_download_file_allows_real_gguf(tmp_path: Path):
    user_id = "user-1"
    model_dir = tmp_path / "models" / user_id / "bundle"
    model_dir.mkdir(parents=True)
    gguf = model_dir / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"gguf")

    resolved = pick_user_download_file(
        tmp_path,
        user_id,
        model_dir,
        relative_name="model-Q4_K_M.gguf",
    )
    assert resolved == gguf.resolve()
