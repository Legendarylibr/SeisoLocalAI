"""Tests for multi-quant regression study helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from seiso.experiments.hf_deploy_regression import summarize_hf_deploy_report
from seiso.experiments.quant_regression import (
    QuantRegressionReport,
    QuantRegressionRow,
    build_eval_route_prompt_library,
    build_llama_cpp_router_routes,
    build_route_catalog,
    format_report_table,
    gguf_route_bits,
    llama_cpp_ready,
    resolve_llama_cpp_python_shim,
    summarize_route_report,
)
from seiso.rl_quant.bootstrap import require_adaptive_quant
from seiso.rl_quant.config_builder import build_framework_config
from seiso.training.config import TrainConfig


def test_gguf_route_bits_mapping():
    assert gguf_route_bits("q4_k_m") == 4
    assert gguf_route_bits("q8_0") == 8
    assert gguf_route_bits("f16") == 16


def test_build_llama_cpp_router_routes(tmp_path: Path):
    gguf = tmp_path / "model-q4_k_m.gguf"
    gguf.write_bytes(b"fake")
    routes = build_llama_cpp_router_routes({"q4_k_m": gguf})
    assert routes == [f"llama_cpp:{gguf.resolve()}@q4"]


def test_build_route_catalog_requires_two_routes(tmp_path: Path):
    require_adaptive_quant()
    one = tmp_path / "a.gguf"
    one.write_bytes(b"x" * 128)
    with pytest.raises(ValueError, match="at least two"):
        build_route_catalog({"q8_0": one})


def test_build_route_catalog_from_ggufs(tmp_path: Path):
    require_adaptive_quant()
    q8 = tmp_path / "model-q8_0.gguf"
    f16 = tmp_path / "model-f16.gguf"
    q8.write_bytes(b"x" * 128)
    f16.write_bytes(b"y" * 256)
    catalog = build_route_catalog({"q8_0": q8, "f16": f16})
    assert len(catalog.routes) == 2


def test_summarize_route_report():
    report = {
        "mean_selected_memory_mb": 1800.0,
        "rows": [
            {"route_id": "gguf_q8_0", "reward": 1.0, "perplexity": 10.0},
            {"route_id": "gguf_f16", "reward": 1.2, "perplexity": 9.5},
        ],
        "recommendations": [
            {
                "route_id": "gguf_q8_0",
                "quant_label": "Q8_0",
                "reward_regression": 0.02,
                "perplexity_regression": 0.01,
                "memory_mb": 1800.0,
            }
        ],
    }
    metrics = summarize_route_report(report)
    assert metrics["eval_mean_reward"] == pytest.approx(1.1)
    assert metrics["recommended_quant"] == "Q8_0"
    assert metrics["reward_regression"] == pytest.approx(0.02)


def test_summarize_route_report_ignores_non_finite_metrics():
    report = {
        "rows": [
            {"route_id": "gguf_q4", "reward": "nan", "perplexity": "inf"},
            {"route_id": "gguf_q8", "reward": "bad", "perplexity": None},
        ],
        "recommendations": [
            {
                "route_id": "gguf_q4",
                "quant_label": "Q4_K_M",
                "reward_regression": "bad",
                "perplexity_regression": "inf",
            }
        ],
    }
    metrics = summarize_route_report(report)
    assert metrics["eval_mean_reward"] is None
    assert metrics["eval_mean_perplexity"] is None
    assert metrics["reward_regression"] is None
    assert metrics["perplexity_regression"] is None


def test_summarize_hf_deploy_report_ignores_non_finite_metrics():
    report = {
        "rows": [{"reward": "nan", "perplexity": "inf"}],
        "recommendations": [
            {"route_id": "4bit", "deploy_quant": "4bit", "memory_mb": "bad"}
        ],
    }
    metrics = summarize_hf_deploy_report(report)
    assert metrics["eval_mean_reward"] is None
    assert metrics["eval_mean_perplexity"] is None
    assert metrics["recommended_quant"] == "4bit"
    assert metrics["mean_selected_memory_mb"] is None


def test_format_report_table_includes_errors():
    report = QuantRegressionReport(
        model_id="test/model",
        study_dir="/tmp/study",
        rows=[
            QuantRegressionRow(
                train_quant="4bit",
                checkpoint="/ckpt",
                eval_mean_reward=0.5,
                backend="llama_cpp",
            ),
            QuantRegressionRow(train_quant="8bit", checkpoint="", error="boom"),
        ],
    )
    text = format_report_table(report)
    assert "4bit" in text
    assert "ERROR: boom" in text


def test_build_framework_config_router_routes(tmp_path: Path):
    require_adaptive_quant()
    cfg = build_framework_config(
        job_id="job-r",
        user_id="user-r",
        data_dir=tmp_path,
        payload={
            "preset": "minimal",
            "router_enabled": True,
            "router_routes": ["llama_cpp:/tmp/a.gguf@q4", "llama_cpp:/tmp/b.gguf@q8"],
            "llama_cpp_timeout_s": 600,
            "hardware_modes": ["gpu"],
        },
    )
    assert cfg.router_enabled is True
    assert len(cfg.router_routes) == 2
    assert cfg.llama_cpp_timeout_s == pytest.approx(600.0)
    assert cfg.hardware_modes == ("gpu",)


def test_resolve_llama_cpp_python_shim_exists():
    shim = resolve_llama_cpp_python_shim()
    assert shim is not None
    assert shim.name == "llama_cli_python_shim.py"


def test_llama_cpp_ready_uses_python_shim_when_gpu_available(monkeypatch):
    monkeypatch.setattr(
        "seiso.experiments.quant_regression.resolve_gguf_converter",
        lambda: "/tmp/convert_hf_to_gguf.py",
    )
    monkeypatch.setattr(
        "seiso.experiments.quant_regression.llama_cpp_python_gpu_ready",
        lambda: True,
    )
    monkeypatch.setattr(
        "seiso.experiments.quant_regression.resolve_llama_cpp_binary",
        lambda explicit=None: resolve_llama_cpp_python_shim(),
    )
    assert llama_cpp_ready() is True


def test_build_eval_route_prompt_library_from_metamath(monkeypatch, tmp_path: Path):
    require_adaptive_quant()
    cfg = TrainConfig(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        dataset="meta-math/MetaMathQA",
        dataset_format="alpaca",
        output_dir=tmp_path,
        eval_split_ratio=0.1,
        seed=42,
        extra={"max_samples": 32},
    )
    train_out = tmp_path / "train-4bit"
    train_out.mkdir()
    (train_out / "train_config_snapshot.json").write_text(
        cfg.model_dump_json(), encoding="utf-8"
    )

    class _EvalSplit:
        def __init__(self):
            self._rows = [
                {"query": "What is 2+2?", "response": "4"},
                {"query": "What is 3+3?", "response": "6"},
            ]

        def __len__(self):
            return len(self._rows)

        def __iter__(self):
            return iter(self._rows)

        def __getitem__(self, index):
            return self._rows[index]

    class _Dataset:
        def __init__(self):
            self._rows = [{"query": f"q{i}", "response": f"a{i}"} for i in range(20)]

        def __len__(self):
            return len(self._rows)

        def select(self, indices):
            out = _Dataset()
            out._rows = [self._rows[i] for i in indices]
            return out

        def train_test_split(self, *, test_size, seed):
            del test_size, seed
            return {"test": _EvalSplit()}

    monkeypatch.setattr(
        "seiso.training.datasets.load_training_dataset",
        lambda *args, **kwargs: _Dataset(),
    )

    class _Tok:
        def apply_chat_template(
            self, messages, *, tokenize=False, add_generation_prompt=False
        ):
            del tokenize
            parts = [f"{m['role']}: {m['content']}" for m in messages]
            if add_generation_prompt:
                parts.append("assistant:")
            return "\n".join(parts)

    transformers_stub = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: _Tok()
        )
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)

    prompts = build_eval_route_prompt_library(train_out, cfg, max_prompts=4)
    assert len(prompts) == 2
    assert prompts[0].domain == "math"
    assert "2+2" in prompts[0].text
