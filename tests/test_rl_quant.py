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
from seiso.rl_quant.presets import rl_quant_presets_response
from seiso.rl_quant.recommendation import recommendation_to_gguf_quants


def test_rl_quant_presets_response_includes_hints():
    payload = rl_quant_presets_response()
    assert len(payload["presets"]) >= 3
    assert payload["preset_hints"]["minimal"]
    assert payload["reward_weights_help"]["gamma_perplexity"]
    assert payload["auto_sweep_help"]["auto_sweep"]


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
        payload={"preset": "minimal", "training_episodes": 16, "evaluation_episodes": 4},
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
            "write_research_report": False,
            "auto_sweep": False,
        },
        on_log=lambda _m: None,
    )
    assert result.get("output_dir")
    assert Path(str(result["output_dir"])).is_dir()
