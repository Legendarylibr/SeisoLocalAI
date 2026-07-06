"""Tests for local artifact completeness checks."""

from __future__ import annotations

from forge.services.artifact_integrity import inventory_gguf_is_complete


def test_inventory_gguf_uses_size_bytes_when_hub_lookup_fails(tmp_path):
    path = tmp_path / "model-Q4_K_M.gguf"
    path.write_bytes(b"partial")
    row = {
        "path": str(path),
        "format": "gguf",
        "size_bytes": 10_000,
        "source": "hf:org/Model",
    }
    metadata = {
        "repo_id": "org/Model",
        "gguf_repo": "mirror/Model-GGUF",
        "gguf_files": ["model-Q4_K_M.gguf"],
    }

    def boom(_repo: str, _filename: str) -> int:
        raise OSError("offline")

    assert inventory_gguf_is_complete(row, metadata, size_lookup=boom) is False


def test_inventory_gguf_rejects_hf_without_size_when_hub_offline(tmp_path):
    path = tmp_path / "model-Q4_K_M.gguf"
    path.write_bytes(b"partial")
    row = {
        "path": str(path),
        "format": "gguf",
        "size_bytes": 0,
        "source": "hf:org/Model",
    }
    metadata = {
        "repo_id": "org/Model",
        "gguf_repo": "mirror/Model-GGUF",
        "gguf_files": ["model-Q4_K_M.gguf"],
    }

    def boom(_repo: str, _filename: str) -> int:
        raise OSError("offline")

    assert inventory_gguf_is_complete(row, metadata, size_lookup=boom) is False
