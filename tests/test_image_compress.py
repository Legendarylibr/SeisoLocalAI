"""Tests for Stable Diffusion image compression integration."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_image_compress_bootstrap_imports():
    from seiso.image_compress.bootstrap import ensure_sd_compress_importable, vendor_root

    root = ensure_sd_compress_importable()
    assert root == vendor_root()
    assert (root / "sd_compress" / "pipeline.py").is_file()


def test_image_compress_config_builder_smoke(tmp_path: Path):
    from seiso.image_compress.config_builder import PRESETS, STAGE_ORDER, build_pipeline_config

    cfg = build_pipeline_config(
        job_id="test-job",
        user_id="user-1",
        data_dir=tmp_path,
        payload={"preset": "smoke"},
    )

    assert cfg["preset"] == "smoke"
    assert cfg["stages"] == PRESETS["smoke"]["stages"]
    assert all(stage in STAGE_ORDER for stage in cfg["stages"])
    assert Path(cfg["output_root"]).is_dir()
    assert cfg["config"].base_model == "runwayml/stable-diffusion-v1-5"
    assert Path(cfg["config"].data_path).is_file()


def test_image_compress_config_unknown_stage(tmp_path: Path):
    from seiso.image_compress.config_builder import build_pipeline_config

    with pytest.raises(ValueError, match="Unknown pipeline stage"):
        build_pipeline_config(
            job_id="test-job",
            user_id="user-1",
            data_dir=tmp_path,
            payload={"stages": ["not_a_stage"]},
        )
