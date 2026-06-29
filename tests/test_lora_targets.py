"""LoRA target detection should work beyond named model families."""

from __future__ import annotations

import types

from seiso.models.lora_targets import (
    detect_architecture,
    get_lora_target_modules,
    modules_exist_in_model,
)


class _FakeModel:
    def __init__(self, names: list[str], model_type: str | None = None) -> None:
        self.config = types.SimpleNamespace(model_type=model_type) if model_type else None
        self._names = names

    def named_parameters(self):
        return [(name, object()) for name in self._names]


def test_detect_architecture_prefers_config_over_path_name():
    model = _FakeModel([], model_type="qwen2")

    assert detect_architecture("/tmp/not-a-qwen-checkpoint", model) == "qwen2"


def test_unknown_architecture_infers_common_targets_from_parameters():
    model = _FakeModel(
        [
            "transformer.blocks.0.attn.W_pack.weight",
            "transformer.blocks.0.attn.o_proj.weight",
            "transformer.blocks.0.mlp.up_proj.weight",
            "transformer.blocks.0.mlp.down_proj.weight",
        ],
        model_type="brand_new",
    )

    assert get_lora_target_modules("org/brand-new-model", model) == [
        "o_proj",
        "W_pack",
        "up_proj",
        "down_proj",
    ]


def test_modules_exist_in_model_does_not_return_missing_targets():
    model = _FakeModel(["layers.0.attention.out_proj.weight"])

    assert modules_exist_in_model(model, ["q_proj", "k_proj"]) == []
