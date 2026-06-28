"""Training model resolution and HF cache tests."""

from __future__ import annotations

from pathlib import Path

from forge.services.models import list_trainable_models, resolve_training_model_id
from seiso.models.hf_env import configure_hf_hub_cache, resolve_hf_cache_dir


def test_configure_hf_hub_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    cache = configure_hf_hub_cache(tmp_path)
    assert cache == tmp_path / "hf_cache"
    assert cache.is_dir()
    import os

    assert os.environ["HUGGINGFACE_HUB_CACHE"] == str(cache)


def test_resolve_hf_cache_dir_respects_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "custom"))
    assert resolve_hf_cache_dir(tmp_path) == tmp_path / "custom"


def test_resolve_training_model_from_inventory(tmp_path: Path):
    user_id = "user-1"
    model_dir = tmp_path / "models" / user_id / "Llama--3.2-1B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"weights")

    inventory = [
        {
            "id": "m1",
            "name": "Llama-3.2-1B",
            "path": str(model_dir),
            "source": "hf:meta-llama/Llama-3.2-1B-Instruct",
            "format": "safetensors",
            "metadata_json": '{"repo_id": "meta-llama/Llama-3.2-1B-Instruct"}',
        }
    ]
    resolved, local = resolve_training_model_id(
        "meta-llama/Llama-3.2-1B-Instruct",
        data_dir=tmp_path,
        user_id=user_id,
        inventory=inventory,
    )
    assert resolved == str(model_dir.resolve())
    assert local == resolved


def test_resolve_training_model_rejects_gguf_only_repo(tmp_path: Path):
    import pytest

    with pytest.raises(ValueError, match="GGUF-only"):
        resolve_training_model_id(
            "unsloth/gemma-4-E4B-it-GGUF",
            data_dir=tmp_path,
            user_id="user-1",
            inventory=[],
        )


def test_resolve_training_model_skips_gguf_only_cache_marked_safetensors(
    tmp_path: Path,
):
    user_id = "user-1"
    bogus = tmp_path / "models" / user_id / "unsloth--gemma-4-E4B-it-GGUF"
    bogus.mkdir(parents=True)
    (bogus / "config.json").write_text("{}")

    inventory = [
        {
            "id": "bad",
            "name": "gemma-4-E4B-it-GGUF",
            "path": str(bogus),
            "source": "hf:unsloth/gemma-4-E4B-it-GGUF",
            "format": "safetensors",
            "metadata_json": '{"repo_id": "unsloth/gemma-4-E4B-it-GGUF"}',
        }
    ]
    trainable_dir = tmp_path / "models" / user_id / "org--model"
    trainable_dir.mkdir(parents=True)
    (trainable_dir / "model.safetensors").write_bytes(b"weights")
    inventory.append(
        {
            "id": "good",
            "name": "model",
            "path": str(trainable_dir),
            "source": "hf:org/model",
            "format": "safetensors",
            "metadata_json": '{"repo_id": "org/model"}',
        }
    )
    resolved, local = resolve_training_model_id(
        "org/model",
        data_dir=tmp_path,
        user_id=user_id,
        inventory=inventory,
    )
    assert resolved == str(trainable_dir.resolve())
    assert local == resolved


def test_resolve_training_model_hf_fallback(tmp_path: Path):
    resolved, local = resolve_training_model_id(
        "meta-llama/Llama-3.2-1B-Instruct",
        data_dir=tmp_path,
        user_id="user-1",
        inventory=[],
    )
    assert resolved == "meta-llama/Llama-3.2-1B-Instruct"
    assert local is None


def test_list_trainable_models_skips_gguf(tmp_path: Path):
    user_id = "user-1"
    st_dir = tmp_path / "models" / user_id / "trainable"
    st_dir.mkdir(parents=True)
    (st_dir / "config.json").write_text("{}")
    (st_dir / "model.safetensors").write_bytes(b"weights")
    gguf = tmp_path / "models" / user_id / "chat.gguf"
    gguf.write_text("fake")

    inventory = [
        {
            "id": "a",
            "name": "train",
            "path": str(st_dir),
            "source": "hf:org/model",
            "format": "safetensors",
        },
        {
            "id": "b",
            "name": "chat",
            "path": str(gguf),
            "source": "hf:org/model",
            "format": "gguf",
        },
    ]
    models = list_trainable_models(inventory, data_dir=tmp_path, user_id=user_id)
    assert len(models) == 1
    assert models[0]["repo_id"] == "org/model"
