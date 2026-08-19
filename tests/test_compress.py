"""LLM compression integration tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from seiso.compress.bootstrap import (
    bundle_root,
    ensure_codellama_compress_importable,
    require_codellama_compress,
)
from seiso.compress.config_builder import PRESETS, STAGE_ORDER, build_pipeline_config
from seiso.compress.runner import (
    _assert_full_model_dir,
    _resolve_model_dir,
    _trust_remote_code,
)


def test_bundled_source_present():
    root = bundle_root()
    assert (root / "cli.py").is_file()


def test_codellama_compress_importable():
    ensure_codellama_compress_importable()
    require_codellama_compress()
    import seiso.codellama_compress  # noqa: F401


def test_stage_order():
    assert "distill" in STAGE_ORDER
    assert "export" in STAGE_ORDER
    # Quantize must precede evaluate/export so metrics reflect quantized weights.
    assert STAGE_ORDER.index("quantize_gptq") < STAGE_ORDER.index("evaluate")
    assert STAGE_ORDER.index("quantize_awq") < STAGE_ORDER.index("export")


def test_build_pipeline_config_sorts_out_of_order_stages(tmp_path: Path):
    """CMP-ORD: UI/toggle order must not evaluate before quantize."""
    require_codellama_compress()
    cfg = build_pipeline_config(
        job_id="job-ord",
        user_id="user-1",
        data_dir=tmp_path,
        payload={
            "preset": "smoke",
            "stages": ["evaluate", "export", "distill", "quantize_gptq"],
        },
    )
    assert cfg["stages"] == ["distill", "quantize_gptq", "evaluate", "export"]


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
    assert cfg["distill"].trust_remote_code is False
    assert cfg["finetune"].trust_remote_code is False


def test_build_pipeline_config_propagates_trust_remote_code(tmp_path: Path):
    require_codellama_compress()
    cfg = build_pipeline_config(
        job_id="job-1",
        user_id="user-1",
        data_dir=tmp_path,
        payload={"preset": "smoke", "trust_remote_code": True},
    )
    assert cfg["distill"].trust_remote_code is True
    assert cfg["finetune"].trust_remote_code is True
    assert _trust_remote_code(cfg) is True


def test_build_pipeline_config_merges_top_level_distill_knobs(tmp_path: Path):
    require_codellama_compress()
    cfg_path = tmp_path / "compress.json"
    cfg_path.write_text(
        """
{
  "pipeline": {"stages": ["distill", "evaluate"], "distill_steps": 7},
  "distill": {"temperature": 3.5, "alpha": 0.75, "steps": 7}
}
""".strip(),
        encoding="utf-8",
    )
    cfg = build_pipeline_config(
        job_id="job-distill",
        user_id="user-1",
        data_dir=tmp_path,
        payload={"preset": "smoke", "config_file": str(cfg_path)},
    )
    assert cfg["distill"].temperature == pytest.approx(3.5)
    assert cfg["distill"].alpha == pytest.approx(0.75)
    assert cfg["distill"].steps == 7


def test_presets_have_stages():
    for name, preset in PRESETS.items():
        assert preset.get("stages"), f"preset {name} missing stages"


def test_replay_manifest_from_bundled_package(tmp_path: Path):
    """Bundled replay/manifest helpers work without GPU."""
    require_codellama_compress()
    from seiso.codellama_compress.config import DeterminismConfig
    from seiso.codellama_compress.replay import (
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
@pytest.mark.skipif(
    os.environ.get("SEISO_RUN_MODEL_SMOKE_TESTS") != "1",
    reason="requires model download; set SEISO_RUN_MODEL_SMOKE_TESTS=1 to run",
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


def test_forge_pipeline_defaults_are_product_presets():
    from forge.api.routes.compress import CompressStartRequest
    from forge.api.routes.distill_rl import DistillRLStartRequest

    assert DistillRLStartRequest().preset == "reproducible"
    assert CompressStartRequest().preset == "full"


def test_compress_refuses_lora_only_model_dir(tmp_path: Path):
    lora = tmp_path / "adapter"
    lora.mkdir()
    (lora / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="LoRA adapter only"):
        _assert_full_model_dir(lora, "prune")

    cfg = {
        "model_dir": str(lora),
        "stages": ["prune", "finetune"],
    }
    with pytest.raises(ValueError, match="LoRA adapter only"):
        _resolve_model_dir(cfg, tmp_path / "run", "prune")
