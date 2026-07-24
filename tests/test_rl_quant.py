"""RL quantization integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from seiso.rl_quant.bootstrap import (
    bundle_root,
    ensure_adaptive_quant_importable,
    require_adaptive_quant,
)
from seiso.rl_quant.config_builder import build_framework_config
from seiso.rl_quant.presets import (
    get_preset,
    known_preset_ids,
    lookup_preset,
    normalize_preset_id,
    rl_quant_presets_response,
    sweep_grid_for_preset,
)
from seiso.rl_quant.recommendation import recommendation_to_gguf_quants
from seiso.rl_quant.sweep import default_sweep_grid


def test_rl_quant_presets_response_includes_hints():
    payload = rl_quant_presets_response()
    assert len(payload["presets"]) >= 3
    assert "train" in payload["stages"]
    assert payload["help"]["recommend"]
    assert payload["defaults"]["training_episodes"] == 256
    assert payload["preset_hints"]["minimal"]
    assert payload["reward_weights_help"]["gamma_perplexity"]
    assert payload["auto_sweep_help"]["auto_sweep"]


def test_product_preset_registry_is_single_source():
    """API metadata, aliases, and sweep grids share one registry (RP-05)."""
    assert set(known_preset_ids()) == {"minimal", "reproducible", "post_train"}
    assert normalize_preset_id("posttrain") == "post_train"
    assert normalize_preset_id("smoke") == "minimal"
    assert lookup_preset("continuous") is None

    post = get_preset("post_train")
    assert post.prompt_library == "prompts/post_train_library.json"
    assert post.backend == "simulator"
    assert "reward_weights.beta_throughput" in post.sweep_grid

    # Sweep helper must not keep a parallel post_train / smoke grid map.
    assert default_sweep_grid({"preset": "posttrain"}) == sweep_grid_for_preset(
        "post_train"
    )
    assert default_sweep_grid({"preset": "smoke"}) == sweep_grid_for_preset("minimal")


def test_build_framework_config_post_train_uses_product_registry(tmp_path: Path):
    require_adaptive_quant()
    cfg = build_framework_config(
        job_id="job-post",
        user_id="user-1",
        data_dir=tmp_path,
        payload={"preset": "posttrain"},
    )
    assert cfg.backend == "simulator"
    assert cfg.training_backend == "python"
    assert cfg.continuous_training is True
    assert cfg.router_enabled is True
    assert str(cfg.prompt_library_path).endswith("post_train_library.json")


def test_bundled_source_present():
    root = bundle_root()
    assert (root / "research_pipeline.py").is_file()


def test_adaptive_quant_importable():
    ensure_adaptive_quant_importable()
    require_adaptive_quant()
    import seiso.adaptive_quant  # noqa: F401


def test_build_framework_config_minimal(tmp_path: Path):
    require_adaptive_quant()
    cfg = build_framework_config(
        job_id="job-1",
        user_id="user-1",
        data_dir=tmp_path,
        payload={
            "preset": "minimal",
            "training_episodes": 16,
            "evaluation_episodes": 4,
        },
    )
    assert cfg.training_episodes == 16
    assert cfg.backend == "simulator"
    assert str(tmp_path / "rl_quant" / "user-1" / "job-1") in cfg.artifacts.outputs_dir


def test_recommendation_to_gguf_quants_bitwidth():
    rec = {"recommended_quant": {"signature": "mode=fixed;base=4;group=32"}}
    assert recommendation_to_gguf_quants(rec) == ["q4_k_m"]


def test_recommendation_to_gguf_quants_quant_type():
    rec = {"decision": {"deploy": "fixed", "quant_type": "Q8_0"}}
    assert recommendation_to_gguf_quants(rec) == ["q8_0"]


@pytest.mark.slow
def test_run_smoke_pipeline(tmp_path: Path):
    """End-to-end simulator smoke (research-grade reproducible preset, tiny episode count)."""
    from seiso.rl_quant.runner import run_rl_quant_job

    result = run_rl_quant_job(
        job_id="smoke",
        user_id="tester",
        data_dir=tmp_path,
        payload={
            "preset": "minimal",
            "training_episodes": 24,
            "evaluation_episodes": 6,
            "auto_sweep": False,
        },
        on_log=lambda _m: None,
    )
    assert result.get("output_dir")
    assert Path(str(result["output_dir"])).is_dir()
