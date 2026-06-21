"""Shared provenance helpers for reproducible research pipelines."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

from seiso.security.deps import sha256_file


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


def git_commit_optional() -> str | None:
    import subprocess

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
            manifest[rel] = sha256_file(path, max_bytes=512 * 1024 * 1024)
        except (OSError, ValueError):
            manifest[rel] = "error"
        count += 1
    return manifest
