"""Tests for HF Hub auth when HF_HOME is relocated."""

from __future__ import annotations

import os
from pathlib import Path


def test_configure_hf_hub_auth_mirrors_cli_token_into_relocated_hf_home(
    monkeypatch, tmp_path: Path
):
    from seiso.models.hf_env import configure_hf_hub_auth, configure_hf_hub_cache

    cli_token = "hf_test_token_123"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "seiso_hf_home"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    home = Path(os.environ["HOME"])
    hf_cache = home / ".cache" / "huggingface"
    hf_cache.mkdir(parents=True)
    (hf_cache / "token").write_text(cli_token + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "huggingface_hub.HfApi.whoami",
        lambda self, token=None: {"name": "test"},
    )

    configure_hf_hub_cache(tmp_path / "data")
    resolved = configure_hf_hub_auth()

    assert resolved == cli_token
    assert os.environ.get("HUGGING_FACE_HUB_TOKEN") == cli_token
    assert (Path(os.environ["HF_HOME"]) / "token").read_text(
        encoding="utf-8"
    ).strip() == cli_token
