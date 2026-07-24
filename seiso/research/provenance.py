"""Shared provenance helpers for reproducible research pipelines."""

from __future__ import annotations

import contextlib
import json
import os
import warnings
from pathlib import Path
from typing import Any

from seiso.security.deps import sha256_file

_ATTENTION_DETERMINISM_WARNINGS = (
    "Memory Efficient attention defaults to a non-deterministic algorithm",
    "Flash Attention defaults to a non-deterministic algorithm",
    "Flash Attention backward for head dim",
)


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
            for fragment in _ATTENTION_DETERMINISM_WARNINGS:
                warnings.filterwarnings(
                    "ignore",
                    message=fragment,
                    category=UserWarning,
                )
    except ImportError:
        pass


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _jsonable(obj: Any) -> Any:
    """Normalize nested configs so dataclass/dict fingerprints agree after reload."""
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict

        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def content_fingerprint(payload: Any) -> str:
    """Stable SHA-256 fingerprint for JSON-serializable config snapshots."""
    import hashlib

    blob = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def manifest_common_fields(*, config_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    """Shared provenance fields for seiso_manifest / study manifests."""
    from datetime import datetime, timezone

    fields: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_optional(),
    }
    if config_snapshot is not None:
        fields["config_fingerprint"] = content_fingerprint(config_snapshot)
    return fields


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


def directory_checksum_manifest(
    root: Path,
    *,
    max_files: int = 200,
    max_file_bytes: int | None = 512 * 1024 * 1024,
) -> dict[str, str]:
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
            if max_file_bytes is not None and path.stat().st_size > max_file_bytes:
                manifest[rel] = "skipped-large-file"
            else:
                manifest[rel] = sha256_file(path, max_bytes=max_file_bytes)
        except (OSError, ValueError):
            manifest[rel] = "error"
        count += 1
    return manifest
