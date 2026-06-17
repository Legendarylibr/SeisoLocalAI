"""Tests for per-user path policy with inventory symlinks."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.services.user_paths import assert_user_path
from seiso.security import SecurityError


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
