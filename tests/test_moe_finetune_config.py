import sys
from types import SimpleNamespace

from seiso.models.seiso_model import SeisoModel
from seiso.training.recommendations import recommend_training_config


def test_moe_recommendations_are_router_safe_and_report_both_sizes():
    profile = {
        "backend": "cpu",
        "gpus": [],
        "ram_gb": 64,
        "tier": "cpu_only",
        "tier_label": "CPU",
    }

    result = recommend_training_config(
        profile,
        model_id="/models/Qwen3-30B-A3B",
    )

    assert result["is_moe"] is True
    assert result["total_params_b"] == 30.0
    assert result["active_params_b"] == 3.0
    assert result["config"]["batch_size"] == 1
    # Hardware caps must key off resident totals (~30B), not active (~3B).
    assert result["config"]["gradient_accumulation_steps"] >= 32
    assert result["config"]["max_seq_length"] <= 1024
    assert result["config"]["gradient_checkpointing"] is True
    assert any("router" in note.lower() for note in result["notes"])


def test_moe_lora_excludes_and_freezes_router(monkeypatch):
    class Parameter:
        def __init__(self) -> None:
            self.requires_grad = True

        def requires_grad_(self, enabled: bool):
            self.requires_grad = enabled
            return self

    class LoraConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class TaskType:
        CAUSAL_LM = "causal-lm"

    router = Parameter()
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen2_moe", num_experts=8, use_cache=True),
        named_parameters=lambda: [
            ("model.layers.0.mlp.gate.weight", router),
            ("model.layers.0.self_attn.q_proj.weight", Parameter()),
        ],
    )
    fake_peft = SimpleNamespace(
        LoraConfig=LoraConfig,
        TaskType=TaskType,
        get_peft_model=lambda _model, config: config,
    )
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setattr(
        "seiso.models.seiso_model.resolve_lora_target_modules",
        lambda *_args, **_kwargs: ["q_proj"],
    )

    result = SeisoModel.attach_lora(
        model,
        model_id="Qwen/Qwen2-MoE",
        use_gradient_checkpointing=False,
        freeze_moe_router=True,
    )

    assert router.requires_grad is False
    assert result.kwargs["exclude_modules"] == [
        "gate",
        "router",
        "wgate",
        "coefficient",
    ]
