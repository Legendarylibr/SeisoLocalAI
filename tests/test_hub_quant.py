"""Tests for native hub-quant model helpers."""

from __future__ import annotations

from seiso.models.hub_quant import (
    active_params_from_config,
    infer_active_params_b,
    is_hub_model_id,
    is_native_hub_quant_model,
    native_quant_training_block_reason,
    needs_tight_vram_training,
)


class _QuantConfig:
    quantization_config = {"quant_method": "mxfp4"}


class _Fp8Config:
    quantization_config = {"quant_method": "fp8"}


class _MoeConfig:
    num_parameters = 20_000_000_000
    num_local_experts = 32
    num_experts_per_tok = 2
    model_type = "moe"


def test_is_hub_model_id():
    assert is_hub_model_id("org/model")
    assert not is_hub_model_id("/tmp/model")
    assert not is_hub_model_id("local-dir")


def test_is_native_hub_quant_model_from_config():
    assert is_native_hub_quant_model("org/model", config=_QuantConfig(), peek=False)


def test_is_native_hub_quant_model_without_config():
    assert not is_native_hub_quant_model("org/model", peek=False)


def test_active_params_from_moe_config():
    params = active_params_from_config(_MoeConfig())
    assert params == 1.25


def test_infer_active_params_dense_model():
    params = infer_active_params_b("Qwen/Qwen2.5-3B-Instruct")
    assert params >= 2.5


def test_native_quant_training_block_reason_fp8():
    reason = native_quant_training_block_reason("org/model", config=_Fp8Config())
    assert reason is not None
    assert "FP8" in reason


def test_native_quant_training_block_reason_mxfp4():
    assert native_quant_training_block_reason("org/model", config=_QuantConfig()) is None


def test_needs_tight_vram_for_native_quant(monkeypatch):
    monkeypatch.setattr(
        "seiso.models.hub_quant.is_native_hub_quant_model",
        lambda model_id, **kwargs: True,
    )
    assert needs_tight_vram_training("org/model")
