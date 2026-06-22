"""Tests for trainable Hub mirror resolution."""

from __future__ import annotations

import pytest


def test_resolve_trainable_hub_id_uses_mirror_when_official_gated(monkeypatch):
    from seiso.models import trainable_mirrors

    def fake_download(repo_id, filename, token=None):
        if repo_id == "google/gemma-3-12b-it":
            raise OSError("403 Client Error: gated repo")
        return "/tmp/config.json"

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    resolved, note = trainable_mirrors.resolve_trainable_hub_id(
        "google/gemma-3-12b-it",
        token="hf_test",
    )
    assert resolved == "unsloth/gemma-3-12b-it"
    assert note and "mirror" in note.lower()


def test_resolve_trainable_hub_id_does_not_mirror_on_transient_error(monkeypatch):
    from seiso.models import trainable_mirrors

    def fake_download(repo_id, filename, token=None):
        raise TimeoutError("timed out connecting to huggingface.co")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    with pytest.raises(ValueError, match="timed out|timeout|network"):
        trainable_mirrors.resolve_trainable_hub_id(
            "google/gemma-3-12b-it",
            token="hf_test",
        )


def test_is_gated_hub_error_detects_403_message():
    from seiso.models.hub_errors import is_gated_hub_error

    assert is_gated_hub_error(OSError("403 Forbidden: gated"))
    assert not is_gated_hub_error(TimeoutError("connection timed out"))
