"""Tests for model-aware context window limits."""

from __future__ import annotations

import json
from pathlib import Path

from seiso.inference.context_limits import (
    context_window_presets,
    effective_context_ceiling,
    hf_config_context_length,
    resolve_model_context_ceiling,
)


def test_hf_config_context_length_reads_max_position(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"max_position_embeddings": 131072}),
        encoding="utf-8",
    )
    assert hf_config_context_length(str(model_dir)) == 131072


def test_resolve_model_context_ceiling_from_name():
    assert (
        resolve_model_context_ceiling("/models/foo-32k.gguf", model_format="gguf")
        == 32768
    )
    assert (
        resolve_model_context_ceiling("/models/foo.gguf", model_name="Llama-3.1-128k")
        == 131072
    )


def test_resolve_model_context_ceiling_defaults_unknown():
    assert resolve_model_context_ceiling(None) == 8192


def test_context_window_presets_includes_effective_max():
    presets = context_window_presets(49152)
    assert 32768 in presets
    assert 49152 in presets
    assert 65536 not in presets


def test_effective_context_ceiling_clamps_to_vram(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 4096)
    model_dir = tmp_path / "big"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"max_position_embeddings": 131072}),
        encoding="utf-8",
    )
    ceiling = effective_context_ceiling(str(model_dir), model_format="safetensors")
    # (headroom - overhead) * 5, rounded down to 512-token steps
    assert ceiling == 18944
    assert ceiling >= 2048


def test_gguf_context_length_reads_metadata(monkeypatch, tmp_path: Path):
    import struct

    from seiso.inference.backends import clear_gguf_caches, gguf_context_length

    clear_gguf_caches()
    gguf = tmp_path / "model.gguf"
    key = b"llama.context_length"
    value = struct.pack("<I", 32768)
    gguf.write_bytes(
        b"GGUF"
        + struct.pack("<IQQ", 3, 0, 1)
        + struct.pack("<Q", len(key))
        + key
        + struct.pack("<I", 4)
        + value
    )
    assert gguf_context_length(str(gguf)) == 32768
