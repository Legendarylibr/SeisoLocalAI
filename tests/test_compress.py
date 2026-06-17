"""Code Llama compression integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from seiso.compress.bootstrap import (
    ensure_codellama_compress_importable,
    require_codellama_compress,
    vendor_root,
)
from seiso.compress.config_builder import PRESETS, STAGE_ORDER, build_pipeline_config


def test_vendor_tree_present():
    root = vendor_root()
    assert (root / "src" / "codellama_compress" / "cli.py").is_file()
    assert (root / "LICENSE").is_file()


def test_codellama_compress_importable():
    ensure_codellama_compress_importable()
    require_codellama_compress()
    import codellama_compress  # noqa: F401


def test_stage_order():
    assert "distill" in STAGE_ORDER
    assert "export" in STAGE_ORDER


def test_build_pipeline_config_smoke(tmp_path: Path):
    require_codellama_compress()
    cfg = build_pipeline_config(
        job_id="job-1",
        user_id="user-1",
        data_dir=tmp_path,
        payload={"preset": "smoke"},
    )
    assert cfg["preset"] == "smoke"
    assert "distill" in cfg["stages"]
    assert str(tmp_path / "compress" / "user-1" / "job-1") in cfg["output_root"]


def test_presets_have_stages():
    for name, preset in PRESETS.items():
        assert preset.get("stages"), f"preset {name} missing stages"


def test_replay_manifest_from_vendor(tmp_path: Path):
    """Vendor replay/manifest helpers work without GPU."""
    require_codellama_compress()
    from codellama_compress.config import DeterminismConfig
    from codellama_compress.replay import (
        append_artifact_record,
        content_fingerprint,
        init_manifest,
        verify_manifest,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "distilled"
    artifact.mkdir()
    (artifact / "weights.txt").write_text("hello", encoding="utf-8")

    pipeline = {"dataset": {"seed": 1}, "determinism": DeterminismConfig()}
    effective = {"stage": "distill", **pipeline}
    fp = content_fingerprint(pipeline)
    init_manifest(
        run_dir,
        config_fingerprint=fp,
        pipeline_fingerprint=pipeline,
        determinism=DeterminismConfig(),
        effective_config=effective,
        stage="distill",
    )
    append_artifact_record(run_dir, stage="distill", artifact_path=artifact, role="output")

    ok = verify_manifest(run_dir)
    assert ok["ok"] is True

    (artifact / "weights.txt").write_text("tampered", encoding="utf-8")
    bad = verify_manifest(run_dir)
    assert bad["ok"] is False


@pytest.mark.slow
@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torch"),
    reason="torch not installed",
)
def test_run_smoke_pipeline(tmp_path: Path):
    """End-to-end smoke pipeline with minimal steps (requires torch + GPU/CPU)."""
    from seiso.compress.runner import run_compress_job

    result = run_compress_job(
        job_id="smoke",
        user_id="tester",
        data_dir=tmp_path,
        payload={
            "preset": "smoke",
            "distill_steps": 1,
            "finetune_steps": 1,
            "max_train_samples": 4,
            "calibration_samples": 4,
        },
        on_log=lambda _m: None,
    )
    assert result.get("run_dir")
    assert Path(str(result["run_dir"])).is_dir()
    manifest = result.get("manifest") or {}
    assert manifest.get("ok") is True
