"""Attention backend resolution and SDPA enablement."""

from __future__ import annotations


def test_resolve_attention_env_override(monkeypatch):
    from seiso.kernels import attention as attn

    attn.resolve_attention_implementation.cache_clear()
    monkeypatch.setenv("SEISO_ATTN_IMPLEMENTATION", "sdpa")
    assert attn.resolve_attention_implementation() == "sdpa"
    attn.resolve_attention_implementation.cache_clear()
    monkeypatch.setenv("SEISO_ATTN_IMPLEMENTATION", "fa2")
    assert attn.resolve_attention_implementation() == "flash_attention_2"
    attn.resolve_attention_implementation.cache_clear()
    monkeypatch.delenv("SEISO_ATTN_IMPLEMENTATION", raising=False)


def test_attention_doctor_lines_nonempty():
    from seiso.kernels.attention import attention_doctor_lines

    lines = attention_doctor_lines()
    assert lines
    assert lines[0].startswith("attention:")


def test_training_defaults_include_attention_and_low_vram_keys():
    from seiso.hardware.training import training_defaults

    defaults = training_defaults({"ram_gb": 32, "backend": "cpu", "gpus": []})
    assert "use_fused_ce" in defaults
    assert "kernel_low_vram" in defaults
    assert "attn_implementation" in defaults
    assert "packing" in defaults


def test_platform_caps_exposes_attention():
    from seiso.training.platform_caps import training_capabilities

    training_capabilities.cache_clear()
    caps = training_capabilities()
    assert "attn_implementation" in caps
    assert "flash_attn_available" in caps
    assert "recommend_sequence_packing" in caps
