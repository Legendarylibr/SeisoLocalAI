"""Tests for external GPU VRAM contention detection."""

from __future__ import annotations

import logging

from seiso.hardware import vram_processes as vp


def test_parse_nvidia_smi_process_csv():
    stdout = "60526, /app/.venv/bin/python3, 8334\n5874, firefox, 188"
    parsed = vp._parse_nvidia_smi_process_csv(stdout)
    assert len(parsed) == 2
    assert parsed[0].pid == 60526
    assert parsed[0].used_mb == 8334
    assert "python3" in parsed[0].process_name


def test_external_gpu_compute_processes_excludes_current(monkeypatch):
    monkeypatch.setattr(
        vp,
        "query_gpu_compute_processes",
        lambda: [
            vp.GpuMemoryProcess(pid=111, process_name="python3", used_mb=8000),
            vp.GpuMemoryProcess(pid=222, process_name="forge", used_mb=256),
        ],
    )
    monkeypatch.setattr(vp.os, "getpid", lambda: 222)

    external = vp.external_gpu_compute_processes()
    assert len(external) == 1
    assert external[0].pid == 111


def test_vram_contention_summary_flags_when_model_cannot_fit(monkeypatch):
    monkeypatch.setattr(
        vp,
        "external_gpu_compute_processes",
        lambda **kwargs: [
            vp.GpuMemoryProcess(pid=111, process_name="python3", used_mb=8200),
        ],
    )
    monkeypatch.setattr(vp, "_safe_free_vram_mb", lambda: 4000)

    summary = vp.vram_contention_summary(model_est_mb=12000)
    assert summary["external_vram_mb"] == 8200
    assert summary["free_vram_mb"] == 4000
    assert summary["model_est_mb"] == 12000
    assert summary["contended"] is True
    assert summary["processes"][0]["name"] == "python3"


def test_vram_contention_summary_not_contended_when_model_fits(monkeypatch):
    monkeypatch.setattr(
        vp,
        "external_gpu_compute_processes",
        lambda **kwargs: [
            vp.GpuMemoryProcess(pid=111, process_name="python3", used_mb=1024),
        ],
    )
    monkeypatch.setattr(vp, "_safe_free_vram_mb", lambda: 20000)

    summary = vp.vram_contention_summary(model_est_mb=12000)
    assert summary["contended"] is False


def test_warn_vram_contention_interprets_loaded_model(monkeypatch, caplog):
    monkeypatch.setattr(
        vp,
        "vram_contention_summary",
        lambda **kwargs: {
            "external_vram_mb": 8192,
            "free_vram_mb": 4000,
            "model_est_mb": kwargs.get("model_est_mb", 0),
            "contended": True,
            "processes": [{"pid": 111, "name": "python3", "used_mb": 8192}],
        },
    )

    with caplog.at_level(logging.WARNING):
        result = vp.warn_vram_contention(
            model_est_mb=17000,
            model_name="Qwen3.6-27B-UD-Q4_K_XL.gguf",
            context="model load",
        )

    assert result is not None
    assert "GPU VRAM contention" in caplog.text
    assert "python3" in caplog.text
    assert "continues without waiting" in caplog.text


def test_warn_vram_contention_skips_when_headroom_fits(monkeypatch, caplog):
    monkeypatch.setattr(
        vp,
        "vram_contention_summary",
        lambda **kwargs: {
            "external_vram_mb": 2048,
            "free_vram_mb": 20000,
            "model_est_mb": kwargs.get("model_est_mb", 0),
            "contended": False,
            "processes": [{"pid": 111, "name": "python3", "used_mb": 2048}],
        },
    )

    with caplog.at_level(logging.WARNING):
        result = vp.warn_vram_contention(model_est_mb=12000, model_name="mid.gguf")

    assert result is None
    assert "GPU VRAM contention" not in caplog.text


def test_warn_vram_contention_skips_without_model(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        result = vp.warn_vram_contention(context="startup")

    assert result is None
    assert "GPU VRAM contention" not in caplog.text


def test_warn_vram_contention_never_raises(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("nvidia-smi hung")

    monkeypatch.setattr(vp, "vram_contention_summary", boom)
    assert vp.warn_vram_contention(model_est_mb=12000, model_name="x.gguf") is None


def test_log_vram_contention_at_startup_is_noop():
    assert vp.log_vram_contention_at_startup() is None


def test_warn_before_model_load_interprets_path(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_warn(**kwargs):
        calls.append(kwargs)
        return {"contended": True}

    monkeypatch.setattr(vp, "warn_vram_contention", fake_warn)
    result = vp.warn_before_model_load(model_path="/models/mid.gguf", est_mb=4500)
    assert result is not None
    assert calls[0]["model_est_mb"] == 4500
    assert calls[0]["model_path"] == "/models/mid.gguf"
