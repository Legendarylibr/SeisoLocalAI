"""LoRA target detection should work beyond named model families."""

from __future__ import annotations

import types

import pytest

from seiso.models.lora_targets import (
    detect_architecture,
    get_lora_target_modules,
    has_multimodal_language_model_backbone,
    infer_lora_target_modules_from_module_tree,
    modules_exist_in_model,
    resolve_lora_target_modules,
)


class _FakeModel:
    def __init__(
        self,
        names: list[str],
        model_type: str | None = None,
        *,
        modules: list[tuple[str, object]] | None = None,
    ) -> None:
        self.config = (
            types.SimpleNamespace(model_type=model_type) if model_type else None
        )
        self._names = names
        self._modules = modules or []

    def named_parameters(self):
        return [(name, object()) for name in self._names]

    def named_modules(self):
        yield "", self
        for name in self._names:
            yield name.rsplit(".", 1)[0] if name.endswith(".weight") else name, self
        for name, module in self._modules:
            yield name, module


class Linear:
    pass


class _Other:
    pass


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


def test_multimodal_language_model_targets_use_language_model_regex():
    model = _FakeModel(
        [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
        ],
        model_type="gemma4",
        modules=[
            (
                "model.language_model.layers.0.self_attn.q_proj",
                Linear(),
            ),
            (
                "model.vision_tower.encoder.layers.0.self_attn.q_proj",
                _Other(),
            ),
        ],
    )

    assert has_multimodal_language_model_backbone(model) is True
    assert detect_architecture("google/gemma-4-E4B-it", model) == "gemma"
    assert get_lora_target_modules("google/gemma-4-E4B-it", model) == (
        r".*language_model\..*\.(q_proj|v_proj)"
    )
    assert modules_exist_in_model(
        model, r".*language_model\..*\.(q_proj|v_proj)"
    ) == (r".*language_model\..*\.(q_proj|v_proj)")


def test_generic_multimodal_wrapper_uses_language_model_regex():
    model = _FakeModel(
        [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.vision_model.encoder.layers.0.self_attn.q_proj.weight",
        ],
        model_type="brand_new_multimodal",
        modules=[
            ("model.language_model.layers.0.self_attn.q_proj", Linear()),
            ("model.language_model.layers.0.self_attn.v_proj", Linear()),
            ("model.vision_model.encoder.layers.0.self_attn.q_proj", Linear()),
        ],
    )

    assert resolve_lora_target_modules("org/new-mm-model", model) == (
        r".*language_model\..*\.(q_proj|v_proj)"
    )


def test_resolve_lora_targets_for_llama_models():
    model = _FakeModel(
        [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ],
        model_type="llama",
    )

    assert resolve_lora_target_modules("meta-llama/Llama-3.2-1B", model) == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_resolve_lora_targets_falls_back_to_linear_module_names():
    model = _FakeModel(
        [],
        modules=[
            ("block.foo", Linear()),
            ("block.bar", Linear()),
            ("lm_head", Linear()),
        ],
    )

    assert infer_lora_target_modules_from_module_tree(model) == ["bar", "foo"]
    assert resolve_lora_target_modules("org/custom-linear-model", model) == [
        "bar",
        "foo",
    ]


def test_resolve_lora_targets_falls_back_when_arch_name_mismatches_modules():
    model = _FakeModel(
        [
            "model.layers.0.attn.W_pack.weight",
            "model.layers.0.attn.out_proj.weight",
            "model.layers.0.mlp.fc1.weight",
            "model.layers.0.mlp.fc2.weight",
        ],
        model_type="custom",
    )

    assert resolve_lora_target_modules("meta-llama/Llama-3.2-1B", model) == [
        "W_pack",
        "out_proj",
        "fc1",
        "fc2",
    ]


def test_resolve_lora_targets_honors_explicit_config():
    model = _FakeModel(
        [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
        ]
    )

    assert resolve_lora_target_modules(
        "test/model",
        model,
        configured=["v_proj", "q_proj", "q_proj"],
    ) == ["v_proj", "q_proj"]


def test_resolve_lora_targets_raises_when_nothing_matches():
    model = _FakeModel(["model.embed_tokens.weight"], model_type="custom")

    with pytest.raises(ValueError, match="Could not infer LoRA target modules"):
        resolve_lora_target_modules("org/no-linear-model", model)
