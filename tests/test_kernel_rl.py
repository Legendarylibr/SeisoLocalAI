"""Tests for kernel RL integration across Seiso and adaptive_quant."""

from __future__ import annotations

from pathlib import Path

import pytest

from seiso.rl_quant.bootstrap import require_adaptive_quant


def test_kernel_profiles_and_analytic_speedup():
    from seiso.kernels.tuning import (
        KERNEL_PROFILES,
        analytic_kernel_speedup,
        kernel_profile_by_id,
    )

    assert len(KERNEL_PROFILES) >= 4
    wide = analytic_kernel_speedup(2, hidden_dim=8192, batch_rows=4096)
    narrow = analytic_kernel_speedup(1, hidden_dim=1024, batch_rows=512)
    assert wide > 0.9
    assert narrow > 0.9
    assert kernel_profile_by_id(2)["name"] == "parallax"


def test_kernel_reward_terms():
    require_adaptive_quant()
    from adaptive_quant.configuration import RewardWeights
    from adaptive_quant.reward import compute_weighted_reward

    weights = RewardWeights(theta_kernel_speedup=0.5, iota_kernel_latency=0.01)
    base_metrics = {
        "latency_ms": 100.0,
        "throughput_tps": 50.0,
        "perplexity": 5.0,
        "memory_mb": 1000.0,
    }
    without = compute_weighted_reward(reward_weights=weights, metrics=base_metrics)
    with_kernel = compute_weighted_reward(
        reward_weights=weights,
        metrics={
            **base_metrics,
            "kernel_speedup": 1.2,
            "kernel_latency_ms": 2.0,
        },
    )
    assert with_kernel > without


def test_build_framework_config_kernel_rl(tmp_path):
    require_adaptive_quant()
    from seiso.rl_quant.config_builder import build_framework_config

    cfg = build_framework_config(
        job_id="kernel-job",
        user_id="user-1",
        data_dir=tmp_path,
        payload={
            "preset": "minimal",
            "training_episodes": 8,
            "evaluation_episodes": 4,
            "kernel_rl_enabled": True,
        },
    )
    assert cfg.kernel_rl_enabled is True
    assert cfg.kernel_profile_count >= 4
    assert cfg.state_vector_dim() > 18


def test_policy_kernel_head_act():
    require_adaptive_quant()
    from adaptive_quant.configuration import FrameworkConfig
    from adaptive_quant.environment import AdaptiveQuantizationEnv
    from adaptive_quant.policy import UniversalQuantizationPolicy

    config = FrameworkConfig(
        training_episodes=4,
        evaluation_episodes=2,
        kernel_rl_enabled=True,
        stability_probe_count=1,
        moe_enabled=False,
    )
    env = AdaptiveQuantizationEnv(config, enable_logging=False)
    policy = UniversalQuantizationPolicy(config)
    state = env.reset(episode_index=0)
    decision, trace = policy.act(state, deterministic=True)
    assert "kernel_profile_index" in decision.metadata
    assert any(item.get("head") == "kernel" for item in trace.action_traces)


def test_kernel_integration_merge():
    from seiso.rl_quant.kernel_integration import merge_kernel_metrics

    metrics = {
        "latency_ms": 200.0,
        "throughput_tps": 80.0,
        "perplexity": 6.0,
        "memory_mb": 1200.0,
    }
    kernel = {
        "kernel_speedup": 1.25,
        "kernel_latency_ms": 1.5,
        "kernel_profile_name": "parallax",
        "kernel_benchmark_source": "analytic",
    }
    merged = merge_kernel_metrics(
        metrics, kernel, config=type("C", (), {"kernel_rl_enabled": True})()
    )
    assert merged["latency_ms"] == pytest.approx(160.0)
    assert merged["throughput_tps"] == pytest.approx(100.0)
    assert merged["kernel_profile_name"] == "parallax"


def test_build_framework_config_kernel_flat_keys(tmp_path: Path):
    require_adaptive_quant()
    from seiso.rl_quant.config_builder import build_framework_config

    cfg = build_framework_config(
        job_id="job-k",
        user_id="user-1",
        data_dir=tmp_path,
        payload={
            "preset": "minimal",
            "kernel_rl_enabled": True,
            "kernel_live_benchmark": True,
            "kernel_hidden_dim": 2048,
            "kernel_batch_rows": 1024,
        },
    )
    assert cfg.kernel_rl_enabled is True
    assert cfg.kernel_rl_live_benchmark is True
    assert cfg.kernel_hidden_dim == 2048
    assert cfg.kernel_batch_rows == 1024


@pytest.mark.slow
def test_run_smoke_pipeline_with_kernel_rl(tmp_path):
    from seiso.rl_quant.runner import run_rl_quant_job

    result = run_rl_quant_job(
        job_id="kernel-smoke",
        user_id="tester",
        data_dir=tmp_path,
        payload={
            "preset": "minimal",
            "training_episodes": 20,
            "evaluation_episodes": 6,
            "kernel_rl_enabled": True,
            "write_research_report": False,
            "auto_sweep": False,
        },
        on_log=lambda _m: None,
    )
    assert result.get("output_dir")
    rec = result.get("recommendation") or {}
    if rec:
        decision = rec.get("decision") or rec.get("recommended_quant") or {}
        meta = decision.get("metadata") if isinstance(decision, dict) else {}
        if isinstance(meta, dict) and meta.get("kernel_profile_index") is not None:
            assert meta["kernel_profile_index"] >= 0
