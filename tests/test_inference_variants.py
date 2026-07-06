"""Tests for inference model variant grouping."""

from __future__ import annotations

import pytest

from forge.services.inference_variants import (
    extract_quant_label,
    get_model_variants,
    variant_group_key,
)


def test_extract_quant_label_from_filename():
    assert (
        extract_quant_label(name="model-Q4_K_M.gguf", path="model-Q4_K_M.gguf")
        == "Q4_K_M"
    )
    assert extract_quant_label(name="model", path="weights-IQ4_XS.gguf") == "IQ4_XS"
    assert (
        extract_quant_label(name="model", path="model-Q4_K_XL.gguf")
        == "Q4_K_XL"
    )
    assert (
        extract_quant_label(
            name="model", path="model-q8_0.gguf", metadata={"quant": "Q8_0"}
        )
        == "Q8_0"
    )


def test_variant_group_key_prefers_repo_id():
    opt = {
        "id": "m1",
        "name": "Gemma Q4",
        "source": "hf:google/gemma-2-2b-GGUF",
        "metadata": {"repo_id": "google/gemma-2-2b-GGUF", "gguf_file": "gemma-q4.gguf"},
    }
    assert variant_group_key(opt) == "google/gemma-2-2b-gguf"


@pytest.mark.asyncio
async def test_get_model_variants_groups_local_quants(monkeypatch):
    current = {
        "id": "q4",
        "name": "Model Q4",
        "path": "/models/model-q4.gguf",
        "format": "gguf",
        "backends": ["llamacpp"],
        "metadata": {
            "repo_id": "org/Model-GGUF",
            "gguf_repo": "org/Model-GGUF",
            "gguf_file": "model-q4.gguf",
        },
    }
    sibling = {
        "id": "q8",
        "name": "Model Q8",
        "path": "/models/model-q8.gguf",
        "format": "gguf",
        "backends": ["llamacpp"],
        "metadata": {
            "repo_id": "org/Model-GGUF",
            "gguf_repo": "org/Model-GGUF",
            "gguf_file": "model-q8.gguf",
            "quant": "Q8_0",
        },
    }

    async def fake_get_option(db, user_id, model_id, **kwargs):
        return current if model_id == "q4" else None

    async def fake_list_options(db, user_id, **kwargs):
        return [current, sibling]

    monkeypatch.setattr(
        "forge.services.inference_variants.get_inference_option",
        fake_get_option,
    )
    monkeypatch.setattr(
        "forge.services.inference_variants.list_inference_options",
        fake_list_options,
    )
    monkeypatch.setattr(
        "forge.services.inference_variants._hub_variant_rows",
        lambda *_args, **_kwargs: [
            {
                "quant": "Q3_K_M",
                "gguf_file": "model-q3.gguf",
                "gguf_repo": "org/Model-GGUF",
                "source": "hub",
                "downloaded": False,
            }
        ],
    )

    variants = await get_model_variants(object(), "u1", "q4", hf_token=None)

    assert variants["model_id"] == "q4"
    assert variants["current_quant"] == "Q4"
    assert len(variants["local_variants"]) == 2
    assert variants["local_variants"][0]["selected"] is True
    assert any(
        row["quant"] == "Q3_K_M" and not row["downloaded"]
        for row in variants["hub_variants"]
    )
