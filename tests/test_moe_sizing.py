from types import SimpleNamespace

from seiso.memory.estimates import (
    estimate_gguf_download_bytes,
    estimate_moe_resident_vram_gb,
)
from seiso.models.moe_sizing import sizing_from_config, sizing_from_reference


def test_named_moe_separates_total_and_active_parameters():
    sizing = sizing_from_reference("Qwen/Qwen3-30B-A3B")

    assert sizing.is_moe is True
    assert sizing.total_params_b == 30.0
    assert sizing.active_params_b == 3.0
    assert sizing.compute_note == "~3B active/token · ~30B resident"


def test_mixtral_config_computes_active_expert_fraction():
    config = SimpleNamespace(
        model_type="mixtral",
        num_parameters=46_700_000_000,
        num_local_experts=8,
        num_experts_per_tok=2,
    )

    sizing = sizing_from_config(config, model_id="mistralai/Mixtral-8x7B")

    assert sizing.is_moe is True
    assert sizing.total_params_b == 46.7
    assert sizing.active_params_b == 12.9
    assert sizing.experts_total == 8
    assert sizing.experts_per_tok == 2


def test_mixtral_name_uses_known_total_and_active_sizes():
    sizing = sizing_from_reference("mistralai/Mixtral-8x7B-Instruct-v0.1")

    assert sizing.total_params_b == 46.7
    assert sizing.active_params_b == 12.9


def test_dense_model_uses_total_as_active():
    sizing = sizing_from_reference("meta-llama/Llama-3.3-70B")

    assert sizing.is_moe is False
    assert sizing.total_params_b == 70.0
    assert sizing.active_params_b == 70.0


def test_moe_artifact_and_residency_use_total_parameters():
    resident = estimate_moe_resident_vram_gb(
        "30B",
        quant="Q4_K_M",
        repo_id="Qwen/Qwen3-30B-A3B",
    )
    artifact = estimate_gguf_download_bytes(
        "30B",
        quant="Q4_K_M",
        tags=("moe",),
        repo_id="Qwen/Qwen3-30B-A3B",
    )

    assert resident > 15
    assert artifact > 15 * 1024**3


def test_hub_moe_train_vram_uses_resident_totals(monkeypatch):
    from seiso.memory.estimates import estimate_training_vram_gb
    from seiso.memory.protection.path_vram import _hub_model_vram_mb

    monkeypatch.setattr(
        "seiso.models.hub_quant.peek_hub_config",
        lambda _repo_id: SimpleNamespace(
            model_type="qwen2_moe",
            num_parameters=30_000_000_000,
            num_experts=64,
            num_experts_per_tok=8,
        ),
    )
    monkeypatch.setattr(
        "seiso.models.hub_quant.is_native_hub_quant_model",
        lambda *_args, **_kwargs: False,
    )

    train_mb = _hub_model_vram_mb("Qwen/Qwen3-30B-A3B", mode="train")
    expected_mb = int(estimate_training_vram_gb("30B", quant="4bit", repo_id="") * 1024)
    active_mb = int(estimate_training_vram_gb("3B", quant="4bit", repo_id="") * 1024)

    assert train_mb == expected_mb
    assert train_mb > active_mb
