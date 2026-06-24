"""Tests for trainable catalog search and training recommendations."""

from __future__ import annotations

from seiso.models.catalog import search_trainable_catalog
from seiso.models.trainable_snapshot import is_gguf_only_repo_id
from seiso.training.recommendations import recommend_training_config


def test_is_gguf_only_repo_id_detects_mirrors():
    assert is_gguf_only_repo_id("unsloth/gemma-4-E4B-it-GGUF")
    assert is_gguf_only_repo_id("bartowski/Qwen3-4B-GGUF")
    assert not is_gguf_only_repo_id("Qwen/Qwen2.5-0.5B-Instruct")
    assert not is_gguf_only_repo_id("google/gemma-2-2b-it")


def test_search_trainable_catalog_skips_gguf(monkeypatch):
    rows = [
        {
            "id": "unsloth/gemma-4-E4B-it-GGUF",
            "downloads": 1000,
            "createdAt": "2026-01-01T00:00:00.000Z",
            "pipeline_tag": "text-generation",
            "tags": ["gguf", "safetensors", "text-generation"],
        },
        {
            "id": "Qwen/Qwen2.5-0.5B-Instruct",
            "downloads": 900,
            "createdAt": "2026-01-01T00:00:00.000Z",
            "pipeline_tag": "text-generation",
            "tags": ["safetensors", "text-generation", "instruct"],
        },
    ]

    monkeypatch.setattr(
        "seiso.models.catalog._query_trainable_hub_page",
        lambda **kwargs: (rows, None),
    )
    result = search_trainable_catalog("qwen")
    repo_ids = [m["repo_id"] for m in result.models]
    assert "Qwen/Qwen2.5-0.5B-Instruct" in repo_ids
    assert "unsloth/gemma-4-E4B-it-GGUF" not in repo_ids


def test_recommend_training_config_warns_on_gguf():
    profile = {"tier_label": "Workstation", "ram_gb": 64, "backend": "cuda"}
    rec = recommend_training_config(
        profile,
        model_id="unsloth/gemma-4-E4B-it-GGUF",
        dataset="HuggingFaceH4/no_robots",
    )
    assert rec["trainable"] is False
    assert rec["warnings"]
    assert rec["config"]["dataset_format"] == "chat"


def test_recommend_training_config_scales_small_model():
    profile = {
        "tier": "workstation",
        "tier_label": "Workstation",
        "ram_gb": 64,
        "backend": "cuda",
        "gpus": [{"vram_total_mb": 24576, "vram_used_mb": 0}],
    }
    rec = recommend_training_config(
        profile,
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        dataset="tatsu-lab/alpaca",
    )
    assert rec["trainable"] is True
    assert rec["config"]["dataset_format"] == "alpaca"
    assert rec["config"]["batch_size"] >= 1
