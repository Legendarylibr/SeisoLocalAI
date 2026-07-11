"""Tests for persistent llama.cpp load profile cache."""

from __future__ import annotations

from pathlib import Path

from seiso.inference.llama_load_cache import (
    clear_load_profile_cache,
    get_cached_load_profile,
    profile_from_load_kwargs,
    save_cached_load_profile,
)


def test_save_and_get_cached_profile(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path / "seiso-data"))
    clear_load_profile_cache()

    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf-bytes")

    profile = profile_from_load_kwargs(
        {
            "n_ctx": 4096,
            "n_batch": 512,
            "n_ubatch": 128,
            "flash_attn": True,
            "offload_kqv": True,
        },
        layers=-1,
        load_tier="normal",
    )
    save_cached_load_profile(
        str(model), n_ctx=4096, load_tier="normal", profile=profile
    )

    cached = get_cached_load_profile(str(model), n_ctx=4096, load_tier="normal")
    assert cached is not None
    assert cached["n_gpu_layers"] == -1
    assert cached["n_batch"] == 512
    assert cached["flash_attn"] is True

    # Nearby ctx buckets share the same profile key (4k band).
    nearby = get_cached_load_profile(str(model), n_ctx=3500, load_tier="normal")
    assert nearby is not None
    assert nearby["n_gpu_layers"] == -1

    clear_load_profile_cache()
    assert get_cached_load_profile(str(model), n_ctx=4096) is None


def test_cache_invalidates_when_file_changes(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path / "seiso-data"))
    clear_load_profile_cache()

    model = tmp_path / "model.gguf"
    model.write_bytes(b"v1")
    save_cached_load_profile(
        str(model),
        n_ctx=2048,
        load_tier="normal",
        profile={"n_gpu_layers": 20, "n_batch": 256, "load_tier": "normal"},
    )
    assert get_cached_load_profile(str(model), n_ctx=2048) is not None

    model.write_bytes(b"v2-longer-content")
    assert get_cached_load_profile(str(model), n_ctx=2048) is None
