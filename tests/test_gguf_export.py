"""Tests for GGUF quant normalization and converter resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from seiso.export.formats import (
    _resolve_merge_base_model,
    merge_lora_checkpoint,
    validate_lora_checkpoint,
)
from seiso.export.gguf import (
    _export_quants,
    normalize_gguf_quant,
    normalize_gguf_quants,
    resolve_gguf_converter,
)


def test_normalize_gguf_quant_aliases():
    assert normalize_gguf_quant("Q4_K_M") == "q4_k_m"
    assert normalize_gguf_quant("q8-0") == "q8_0"
    assert normalize_gguf_quant("F16") == "f16"


def test_normalize_gguf_quants_deduplicates():
    assert normalize_gguf_quants(["Q4_K_M", "q4_k_m", "q8_0"]) == ["q4_k_m", "q8_0"]


def test_normalize_gguf_quants_fallback():
    assert normalize_gguf_quants([]) == ["q4_k_m"]


def test_resolve_gguf_converter_llama_cpp_dir(monkeypatch, tmp_path: Path):
    script = tmp_path / "convert_hf_to_gguf.py"
    script.write_text("# mock converter\n")
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path))
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/python3" if name == "python3" else None
    )
    cmds = resolve_gguf_converter()
    assert cmds
    assert str(script) in cmds[0][1]


def test_export_quants_reuses_single_f16_intermediate(monkeypatch, tmp_path: Path):
    calls: list[tuple[str, Path]] = []
    quantized: list[tuple[str, Path]] = []

    def fake_convert(_source: Path, dest: Path, outtype: str, _log):
        calls.append((outtype, dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"f16")
        return True

    def fake_quantize(source: Path, dest: Path, quant: str, _log):
        assert source.is_file()
        quantized.append((quant, source))
        dest.write_bytes(quant.encode())
        return True

    monkeypatch.setattr("seiso.export.gguf._convert_hf_dir_direct", fake_convert)
    monkeypatch.setattr("seiso.export.gguf.quantize_gguf_file", fake_quantize)

    results = _export_quants(
        tmp_path / "merged",
        tmp_path / "out",
        ["q4_k_m", "q5_k_m"],
        "model",
        lambda _msg: None,
    )

    assert [call[0] for call in calls] == ["f16"]
    # Quantize workers may finish out of order; results are recorded in request order.
    assert {item[0] for item in quantized} == {"q4_k_m", "q5_k_m"}
    assert quantized[0][1] == quantized[1][1]
    assert list(results) == ["gguf_q4_k_m", "gguf_q5_k_m"]


def test_export_quants_reuses_requested_f16_for_later_k_quant(monkeypatch, tmp_path: Path):
    calls: list[tuple[str, Path]] = []
    quantized: list[tuple[str, Path]] = []

    def fake_convert(_source: Path, dest: Path, outtype: str, _log):
        calls.append((outtype, dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"f16")
        return True

    def fake_quantize(source: Path, dest: Path, quant: str, _log):
        assert source == tmp_path / "out" / "f16" / "model-f16.gguf"
        quantized.append((quant, source))
        dest.write_bytes(quant.encode())
        return True

    monkeypatch.setattr("seiso.export.gguf._convert_hf_dir_direct", fake_convert)
    monkeypatch.setattr("seiso.export.gguf.quantize_gguf_file", fake_quantize)

    results = _export_quants(
        tmp_path / "merged",
        tmp_path / "out",
        ["f16", "q4_k_m"],
        "model",
        lambda _msg: None,
    )

    assert [call[0] for call in calls] == ["f16"]
    assert [item[0] for item in quantized] == ["q4_k_m"]
    assert set(results) == {"gguf_f16", "gguf_q4_k_m"}


def test_validate_lora_checkpoint_requires_weights(tmp_path: Path):
    ckpt = tmp_path / "lora"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="adapter weights"):
        validate_lora_checkpoint(ckpt)


def test_validate_lora_checkpoint_ok(tmp_path: Path):
    ckpt = tmp_path / "lora"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_text("x")
    validate_lora_checkpoint(ckpt)


def test_resolve_merge_base_model_from_manifest(tmp_path: Path):
    base = tmp_path / "base-model"
    base.mkdir()
    (base / "config.json").write_text('{"model_type": "llama"}')

    ckpt = tmp_path / "adapter"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text('{"base_model_name_or_path": "hf/model"}')
    (ckpt / "seiso_manifest.json").write_text(
        f'{{"resolved_model_path": "{base}", "model_id": "hf/model"}}'
    )

    assert _resolve_merge_base_model(ckpt) == str(base.resolve())


def test_resolve_merge_base_model_falls_back_to_adapter_hub_id(tmp_path: Path):
    ckpt = tmp_path / "adapter"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text('{"base_model_name_or_path": "remote/model"}')

    assert _resolve_merge_base_model(ckpt) == "remote/model"


@patch("seiso.export.formats._load_merge_deps")
def test_merge_lora_uses_local_base(mock_deps, tmp_path: Path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text('{"base_model_name_or_path": "remote/model"}')
    (ckpt / "seiso_manifest.json").write_text(
        f'{{"resolved_model_path": "{base}", "model_id": "remote/model"}}'
    )
    (ckpt / "tokenizer_config.json").write_text("{}")

    merged = tmp_path / "merged"
    merged.mkdir()

    deps = mock_deps.return_value
    mock_model = deps.auto_model.from_pretrained.return_value
    deps.peft_model.from_pretrained.return_value.merge_and_unload.return_value = mock_model

    merge_lora_checkpoint(ckpt, merged, lambda _msg: None)
    deps.auto_model.from_pretrained.assert_called_once()
    assert deps.auto_model.from_pretrained.call_args[0][0] == str(base.resolve())
