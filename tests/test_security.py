"""Security path validation tests."""

from pathlib import Path

import pytest

from seiso.security import (
    USER_SCOPED_DATA_ROOTS,
    SecurityError,
    assert_user_scoped_path,
    assert_within,
    safe_join,
    sanitize_filename,
)


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


def test_safe_join_embedded_traversal_blocked(tmp_path: Path):
    """A single segment must not smuggle ../ past safe_join."""
    base = tmp_path / "sandbox"
    (base / "uploads" / "attacker").mkdir(parents=True)
    (base / "knowledge" / "victim").mkdir(parents=True)
    with pytest.raises(SecurityError):
        safe_join(base, "uploads", "attacker/../../knowledge/victim")
    with pytest.raises(SecurityError):
        safe_join(base, "alice/../bob")


def test_safe_join_rejects_symlink_segment(tmp_path: Path):
    """Planted tenant symlink must not redirect safe_join write sinks."""
    base = tmp_path / "sandbox"
    bob = base / "knowledge" / "bob" / "kb1"
    bob.mkdir(parents=True)
    (bob / "index.jsonl").write_text("bob\n", encoding="utf-8")
    alice_link = base / "knowledge" / "alice"
    alice_link.parent.mkdir(parents=True, exist_ok=True)
    alice_link.symlink_to(base / "knowledge" / "bob")
    with pytest.raises(SecurityError, match="Symlink rejected"):
        safe_join(base, "knowledge", "alice", "kb1")


def test_sanitize_filename():
    assert "evil" in sanitize_filename("../../evil")
    assert sanitize_filename("") == "unnamed"
    assert sanitize_filename("My Model v1.safetensors") == "My Model v1.safetensors"


def test_user_scoped_data_roots_cover_tenant_categories():
    """Canonical set used by CLI + Forge; keep categories explicit."""
    expected = {
        "uploads",
        "knowledge",
        "artifacts",
        "sandbox",
        "models",
        "checkpoints",
        "exports",
        "compress",
        "distill_rl",
        "recipes",
    }
    assert expected == USER_SCOPED_DATA_ROOTS
    assert "hf_cache" not in USER_SCOPED_DATA_ROOTS


def test_assert_user_scoped_path_allows_owner_tree(tmp_path: Path):
    user_id = "user-1"
    target = tmp_path / "uploads" / user_id / "data.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    assert assert_user_scoped_path(tmp_path, user_id, target) == target.resolve()


def test_assert_user_scoped_path_rejects_cross_user(tmp_path: Path):
    target = tmp_path / "models" / "other" / "m.gguf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    with pytest.raises(SecurityError, match="Path must be under"):
        assert_user_scoped_path(tmp_path, "user-1", target)


def test_assert_user_scoped_path_rejects_hf_cache_direct(tmp_path: Path):
    """Shared cache is not a user-scoped root (Forge inventory links handle access)."""
    cache = tmp_path / "hf_cache" / "blob.bin"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"x")
    with pytest.raises(SecurityError, match="Access denied to path root"):
        assert_user_scoped_path(tmp_path, "user-1", cache)
