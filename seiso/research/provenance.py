"""Shared provenance helpers for reproducible research pipelines."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path, max_bytes: int = 512 * 1024 * 1024) -> str:
    """Return hex SHA-256 of a file (up to max_bytes)."""
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while chunk := f.read(65536):
            size += len(chunk)
            if size > max_bytes:
                raise ValueError(f"File exceeds hash limit: {path}")
            h.update(chunk)
    return h.hexdigest()


def git_commit_optional() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def pip_freeze_lines(limit: int = 80) -> list[str]:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if out.returncode == 0:
            lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
            return lines[:limit]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return []


def collect_env_report() -> dict[str, Any]:
    import platform

    from seiso.security.hardware_privacy import sanitize_env_report

    raw: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_optional(),
        "pip_freeze": pip_freeze_lines(),
    }
    try:
        import torch

        raw["torch"] = torch.__version__
        raw["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            raw["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        raw["torch"] = None
        raw["cuda_available"] = False
    return sanitize_env_report(raw)


def apply_determinism(seed: int, *, deterministic: bool = True) -> None:
    """Set RNG seeds and optional strict deterministic CUDA mode."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            with contextlib.suppress(Exception):
                torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def directory_checksum_manifest(root: Path, *, max_files: int = 200) -> dict[str, str]:
    """SHA-256 manifest for files under root (relative paths → hex)."""
    manifest: dict[str, str] = {}
    if not root.is_dir():
        return manifest
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if count >= max_files:
            break
        rel = str(path.relative_to(root))
        try:
            manifest[rel] = sha256_file(path)
        except (OSError, ValueError):
            manifest[rel] = "error"
        count += 1
    return manifest


def write_run_provenance(
    output_dir: Path,
    *,
    pipeline: str,
    config: dict[str, Any],
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write seiso_run_provenance.json with config snapshot and environment."""
    payload: dict[str, Any] = {
        "pipeline": pipeline,
        "seed": seed,
        "config": config,
        "environment": collect_env_report(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload["extra"] = extra
    path = output_dir / "seiso_run_provenance.json"
    write_json(path, payload)
    return path
