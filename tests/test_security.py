"""Security path validation tests."""

import pytest
from pathlib import Path

from seiso.security import SecurityError, assert_within, safe_join, sanitize_filename


def test_safe_join_blocks_traversal(tmp_path: Path):
    base = tmp_path / "sandbox"
    base.mkdir()
    with pytest.raises(SecurityError):
        safe_join(base, "..", "etc", "passwd")


def test_safe_join_valid(tmp_path: Path):
    base = tmp_path / "sandbox"
    base.mkdir()
    result = safe_join(base, "models", "llama.gguf")
    assert result.exists() is False
    assert str(result.resolve()).startswith(str(base.resolve()))


def test_assert_within_outside_path(tmp_path: Path):
    base = tmp_path / "data"
    base.mkdir()
    inner = base / "file.txt"
    inner.write_text("ok")
    assert assert_within(base, inner) == inner.resolve()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope")
    with pytest.raises(SecurityError):
        assert_within(base, outside)


def test_assert_within_prefix_bypass_blocked(tmp_path: Path):
    """Paths like sandbox-evil must not pass when base is sandbox."""
    base = tmp_path / "sandbox"
    base.mkdir()
    evil = tmp_path / "sandbox-evil"
    evil.mkdir()
    (evil / "secret.txt").write_text("nope")
    with pytest.raises(SecurityError):
        assert_within(base, evil / "secret.txt")


def test_safe_join_prefix_bypass_blocked(tmp_path: Path):
    base = tmp_path / "data"
    base.mkdir()
    with pytest.raises(SecurityError):
        safe_join(base, "..", "sandbox-evil", "file.txt")


def test_sanitize_filename():
    assert "evil" in sanitize_filename("../../evil")
    assert sanitize_filename("") == "unnamed"
    assert sanitize_filename("My Model v1.safetensors") == "My Model v1.safetensors"
