"""DPO alignment LoRA target resolution should not be model-family locked."""

from __future__ import annotations

import types

import pytest

from seiso.adaptive_quant.llm_alignment.config import DPOSettings
from seiso.adaptive_quant.llm_alignment.model_loading import (
    resolve_alignment_lora_target_modules,
)


class _FakeModel:
    def __init__(self, names: list[str], model_type: str | None = None) -> None:
        self.config = types.SimpleNamespace(model_type=model_type) if model_type else None
        self._names = names

    def named_parameters(self):
        return [(name, object()) for name in self._names]


def test_alignment_lora_targets_fall_back_to_model_parameters():
    settings = DPOSettings(sft_model_path="org/custom-model")
    model = _FakeModel(
        [
            "model.layers.0.attn.W_pack.weight",
            "model.layers.0.attn.out_proj.weight",
            "model.layers.0.mlp.fc1.weight",
            "model.layers.0.mlp.fc2.weight",
        ],
        model_type="custom",
    )

    assert resolve_alignment_lora_target_modules(model, settings) == [
        "W_pack",
        "out_proj",
        "fc1",
        "fc2",
    ]


def test_alignment_lora_targets_keep_existing_configured_matches():
    settings = DPOSettings(
        sft_model_path="org/model",
        lora_target_modules=("dense_h_to_4h", "dense_4h_to_h"),
    )
    model = _FakeModel(
        [
            "transformer.h.0.mlp.dense_h_to_4h.weight",
            "transformer.h.0.mlp.dense_4h_to_h.weight",
        ],
        model_type="gpt_neox",
    )

    assert resolve_alignment_lora_target_modules(model, settings) == [
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]


def test_alignment_lora_targets_fail_when_no_targets_exist():
    settings = DPOSettings(sft_model_path="org/no-linear-model")
    model = _FakeModel(["model.embed_tokens.weight"], model_type="custom")

    with pytest.raises(ValueError, match="Could not infer LoRA target modules"):
        resolve_alignment_lora_target_modules(model, settings)

