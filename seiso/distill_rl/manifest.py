"""Hash-chained manifest and environment provenance for distill-rl runs."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from seiso.distill_rl.config import DistillRLConfig


def pipeline_fingerprint(config: DistillRLConfig) -> dict[str, Any]:
    return config.model_dump(mode="json")


def init_run_manifest(config: DistillRLConfig) -> dict[str, Any]:
    from seiso.compress.bootstrap import require_codellama_compress

    require_codellama_compress()
    from seiso.codellama_compress.replay import content_fingerprint, init_manifest

    fp = content_fingerprint(pipeline_fingerprint(config))
    manifest = cast(
        dict[str, Any],
        init_manifest(
            config.output_root,
            config_fingerprint=fp,
            determinism={"seed": config.seed, "deterministic": config.deterministic},
            effective_config=pipeline_fingerprint(config),
            pipeline_fingerprint=pipeline_fingerprint(config),
            stage=config.stages[0] if config.stages else "distill",
        ),
    )
    manifest["pipeline"] = "distill_rl"
    manifest["job_id"] = config.job_id
    manifest["user_id"] = config.user_id
    manifest["environment"] = collect_environment()
    _write_manifest(config.output_root, manifest)
    return manifest


def append_artifact(
    run_dir: Path,
    *,
    stage: str,
    artifact_path: Path,
    role: str = "output",
    extra: dict[str, Any] | None = None,
) -> None:
    from seiso.compress.bootstrap import require_codellama_compress

    require_codellama_compress()
    from seiso.codellama_compress.replay import append_artifact_record

    append_artifact_record(
        run_dir,
        stage=stage,
        artifact_path=artifact_path,
        role=role,
        extra=extra,
    )


def verify_run_manifest(run_dir: Path) -> dict[str, Any]:
    from seiso.compress.bootstrap import require_codellama_compress

    require_codellama_compress()
    from seiso.codellama_compress.replay import verify_manifest

    return cast(dict[str, Any], verify_manifest(run_dir))


def collect_environment() -> dict[str, str]:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        env["git_commit"] = rev
    except (OSError, subprocess.CalledProcessError):
        env["git_commit"] = ""
    return env


def _write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    from seiso.compress.bootstrap import require_codellama_compress

    require_codellama_compress()
    from seiso.codellama_compress.replay import write_manifest

    write_manifest(run_dir, manifest)
