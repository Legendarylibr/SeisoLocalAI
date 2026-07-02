"""Tests for local artifact completeness checks."""

from __future__ import annotations

import pytest

from forge.services import artifact_integrity


def test_path_has_complete_artifact_short_circuits_unknown_directory_size(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    first = model_dir / "first.gguf"
    second = model_dir / "second.gguf"
    first.write_bytes(b"gguf")
    second.write_bytes(b"more")

    yielded = []

    def fake_iter_matching_files(*_args, **_kwargs):
        yielded.append(first.name)
        yield first
        pytest.fail("unknown-size completeness should stop after first usable file")

    monkeypatch.setattr(artifact_integrity, "iter_matching_files", fake_iter_matching_files)

    assert artifact_integrity.path_has_complete_artifact(model_dir, "gguf", 0)
    assert yielded == [first.name]


def test_inventory_gguf_support_check_uses_metadata_file(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    wanted = model_dir / "model-Q4_K_M.gguf"
    other = model_dir / "model-Q8_0.gguf"
    wanted.write_bytes(b"q4")
    other.write_bytes(b"q8-larger")

    seen: list[str] = []
    monkeypatch.setattr(
        artifact_integrity,
        "gguf_is_supported_by_llamacpp",
        lambda path: seen.append(path) or True,
    )

    assert artifact_integrity.inventory_gguf_is_complete(
        {
            "path": str(model_dir),
            "format": "gguf",
            "source": "hf:org/Model",
        },
        {
            "repo_id": "org/Model",
            "gguf_files": ["model-Q4_K_M.gguf"],
        },
        size_lookup=lambda _repo, _filename: wanted.stat().st_size,
    )
    assert seen == [str(wanted)]
