"""Regression: seiso.hardware must import without circular dependency via seiso.memory."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _clear_platform_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SEISO_LLAMA_") or key in {"SEISO_MEMORY_PROFILE", "SEISO_SKIP_MLX_PROBE"}:
            monkeypatch.delenv(key, raising=False)


def test_hardware_import_without_forge():
    from seiso.hardware import assess_catalog_fit, hardware_profile
    from seiso.memory import apply_platform_memory_profile, memory_profile_label

    profile = hardware_profile()
    assert "ram_gb" in profile
    fit = assess_catalog_fit({"repo_id": "Qwen/Qwen3.6-4B", "params": "4B", "quant": "Q4_K_M"}, profile)
    assert "hardware_fit" in fit
    assert memory_profile_label(profile) in {"low", "balanced"}
    assert "memory_profile" in apply_platform_memory_profile(profile=profile)
