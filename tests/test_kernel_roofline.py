"""Unit tests for Seiso fused-op arithmetic intensity estimates."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from seiso.kernels.roofline import (
    REFERENCE_RIDGE_FLOP_PER_BYTE,
    _SOT_ELIGIBLE_OPS,
    classify_bound,
    dtype_nbytes,
    estimate_cross_entropy,
    estimate_fused_mlp_swiglu,
    estimate_lora_delta,
    estimate_lora_qkv_delta,
    estimate_rms_norm,
    estimate_seiso_fused_ops,
    estimate_swiglu,
    format_roofline_report,
    gemm_flops_bytes,
    is_performance_source_of_truth,
)


def test_dtype_nbytes():
    assert dtype_nbytes("bfloat16") == 2
    assert dtype_nbytes("float16") == 2
    assert dtype_nbytes("float32") == 4
    with pytest.raises(ValueError):
        dtype_nbytes("int8")


def test_reference_ridge_is_300():
    assert REFERENCE_RIDGE_FLOP_PER_BYTE == 300.0


def test_sot_eligible_ops():
    assert _SOT_ELIGIBLE_OPS == {
        "fused_mlp_swiglu",
        "lora_delta",
        "lora_qkv_delta",
    }


def test_classify_bound_sot_only_at_300_plus_gemm_fp16():
    bound, conf, sot, _ = classify_bound(1.0, op="fused_mlp_swiglu", dtype="bfloat16")
    assert bound == "bandwidth" and conf == "heuristic" and sot is False

    bound, conf, sot, _ = classify_bound(150.0, op="fused_mlp_swiglu", dtype="bfloat16")
    assert bound == "mixed" and conf == "heuristic" and sot is False

    bound, conf, sot, _ = classify_bound(299.9, op="fused_mlp_swiglu", dtype="bfloat16")
    assert sot is False and conf == "heuristic"

    bound, conf, sot, _ = classify_bound(300.0, op="fused_mlp_swiglu", dtype="bfloat16")
    assert bound == "compute" and conf == "source_of_truth" and sot is True

    bound, conf, sot, _ = classify_bound(500.0, op="rms_norm+residual", dtype="bfloat16")
    assert sot is False and conf == "heuristic" and bound == "mixed"


def test_float32_never_sot_even_at_high_intensity():
    assert not is_performance_source_of_truth(
        op="fused_mlp_swiglu",
        intensity_flop_per_byte=1000.0,
        dtype="float32",
    )
    bound, conf, sot, note = classify_bound(
        1000.0, op="fused_mlp_swiglu", dtype="float32"
    )
    assert sot is False and conf == "heuristic"
    assert "FP16/BF16" in note


def test_is_performance_source_of_truth_gate():
    assert is_performance_source_of_truth(
        op="fused_mlp_swiglu", intensity_flop_per_byte=300, dtype="bfloat16"
    )
    assert is_performance_source_of_truth(
        op="lora_delta", intensity_flop_per_byte=300, dtype="float16"
    )
    assert not is_performance_source_of_truth(
        op="fused_mlp_swiglu", intensity_flop_per_byte=299.9, dtype="bfloat16"
    )
    assert not is_performance_source_of_truth(
        op="swiglu_elementwise", intensity_flop_per_byte=1000, dtype="bfloat16"
    )


def test_gemm_flops_bytes_classic():
    flops, nbytes = gemm_flops_bytes(m=2, n=3, k=4, elem_bytes=2)
    assert flops == 2 * 2 * 3 * 4
    assert nbytes == (2 * 4 + 4 * 3 + 2 * 3) * 2


def test_fused_mlp_closed_form():
    r, h, mid = 128, 256, 1024
    elem = 2
    est = estimate_fused_mlp_swiglu(
        rows=r, hidden=h, intermediate=mid, dtype="bfloat16"
    )
    f1, b1 = gemm_flops_bytes(m=r, n=mid, k=h, elem_bytes=elem)
    f2, b2 = gemm_flops_bytes(m=r, n=mid, k=h, elem_bytes=elem)
    expect_flops = f1 + f2 + 6.0 * r * mid
    expect_bytes = b1 + b2 + float(r * mid) * elem
    assert est.flops == pytest.approx(expect_flops)
    assert est.bytes_moved == pytest.approx(expect_bytes)
    assert est.traffic_model == "classic_gemm_per_matmul"


def test_lora_qkv_is_three_independent_loras():
    one = estimate_lora_delta(
        rows=256, in_features=512, out_features=512, rank=16, dtype="bfloat16"
    )
    qkv = estimate_lora_qkv_delta(rows=256, hidden=512, rank=16, dtype="bfloat16")
    assert qkv.flops == pytest.approx(3.0 * one.flops)
    assert qkv.bytes_moved == pytest.approx(3.0 * one.bytes_moved)
    assert qkv.intensity_flop_per_byte == pytest.approx(one.intensity_flop_per_byte)
    assert "independent" in qkv.traffic_model


def test_elementwise_ops_are_heuristic_bandwidth():
    rms = estimate_rms_norm(rows=4096, hidden=4096, dtype="bfloat16")
    swiglu = estimate_swiglu(rows=4096, intermediate=4096, dtype="bfloat16")
    assert rms.likely_bound == "bandwidth"
    assert rms.performance_truth is False
    assert rms.traffic_model == "lower_bound_streams"
    assert swiglu.performance_truth is False


def test_fat_mlp_is_performance_sot_compute():
    mlp = estimate_fused_mlp_swiglu(
        rows=4096, hidden=4096, intermediate=16384, dtype="bfloat16"
    )
    assert mlp.intensity_flop_per_byte >= REFERENCE_RIDGE_FLOP_PER_BYTE
    assert mlp.performance_truth is True
    assert mlp.confidence == "source_of_truth"
    assert mlp.likely_bound == "compute"


def test_fat_mlp_float32_not_sot():
    mlp = estimate_fused_mlp_swiglu(
        rows=4096, hidden=4096, intermediate=16384, dtype="float32"
    )
    assert mlp.performance_truth is False
    assert mlp.confidence == "heuristic"


def test_tiny_mlp_not_sot():
    mlp = estimate_fused_mlp_swiglu(
        rows=1, hidden=4096, intermediate=16384, dtype="bfloat16"
    )
    assert mlp.intensity_flop_per_byte < REFERENCE_RIDGE_FLOP_PER_BYTE
    assert mlp.performance_truth is False


def test_lora_typical_rank_not_sot_but_scales():
    low = estimate_lora_delta(
        rows=2048, in_features=4096, out_features=4096, rank=4, dtype="bfloat16"
    )
    high = estimate_lora_delta(
        rows=2048, in_features=4096, out_features=4096, rank=64, dtype="bfloat16"
    )
    assert high.intensity_flop_per_byte > low.intensity_flop_per_byte
    assert low.performance_truth is False


def test_ce_never_sot():
    ce = estimate_cross_entropy(rows=512, vocab=128000, dtype="bfloat16")
    assert ce.performance_truth is False


def test_summary_non_gemm_high_i_message():
    # Synthetic: classify non-GEMM at high I
    _bound, _conf, sot, note = classify_bound(
        500.0, op="swiglu_elementwise", dtype="bfloat16"
    )
    assert sot is False
    assert "not GEMM-family" in note


def test_bundle_and_report_mentions_candidate_not_measured():
    estimates = estimate_seiso_fused_ops(rows=4096, hidden=4096, vocab=32000)
    text = format_roofline_report(estimates)
    assert "300" in text
    assert "candidate" in text.lower()
    assert "not a measured" in text.lower() or "not a measured device" in text.lower()
    assert any(e.performance_truth for e in estimates)


def test_cli_roofline_only_no_gpu():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "seiso.kernels.benchmark",
            "--roofline-only",
            "--rows",
            "4096",
            "--hidden",
            "4096",
            "--vocab",
            "1000",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["scope"] == "seiso_fused_ops_only"
    assert data["reference_ridge_flop_per_byte"] == 300.0
    assert "never blocks training" in data["disclaimer"]
    assert "fused_mlp_swiglu" in data["source_of_truth_ops"]
    for est in data["estimates"]:
        assert "performance_truth" in est
        assert "traffic_model" in est
        if est["performance_truth"]:
            assert est["intensity_flop_per_byte"] >= 300.0
            assert est["confidence"] == "source_of_truth"
            assert est["likely_bound"] == "compute"
            assert est["shape"]["dtype"] in {"float16", "bfloat16"}
