"""Shared provenance helpers for reproducible research pipelines."""

from __future__ import annotations

import contextlib
import hashlib
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
    max_files: int | None = 200,
    max_file_bytes: int | None = 512 * 1024 * 1024,
    always_hash_suffixes: tuple[str, ...] = (),
) -> dict[str, str]:
    """SHA-256 manifest for files under root (relative paths → hex).

    ``always_hash_suffixes`` forces a full hash for matching files even when
    ``max_file_bytes`` would otherwise skip them (export weight tensors).
    ``max_files=None`` hashes every file under ``root``.
    """
    manifest: dict[str, str] = {}
    if not root.is_dir():
        return manifest
    suffixes = tuple(s.lower() for s in always_hash_suffixes)
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if max_files is not None and count >= max_files:
            break
        rel = str(path.relative_to(root))
        force_full = bool(suffixes) and path.suffix.lower() in suffixes
        limit = None if force_full else max_file_bytes
        try:
            if limit is not None and path.stat().st_size > limit:
                manifest[rel] = "skipped-large-file"
            else:
                manifest[rel] = sha256_file(path, max_bytes=limit)
        except (OSError, ValueError):
            manifest[rel] = "error"
        count += 1
    return manifest


ATTESTATION_SCHEMA_V1 = "seiso.provenance.attestation/v1"
ATTESTATION_SCHEMA_V2 = "seiso.provenance.attestation/v2"


def strip_nostr_receipt(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of manifest without the mutable ``nostr`` receipt."""
    return {k: v for k, v in manifest.items() if k != "nostr"}


def manifest_sha256_excluding_nostr(manifest: dict[str, Any]) -> str:
    """SHA-256 of the local manifest with the nostr receipt excluded."""
    return content_fingerprint(strip_nostr_receipt(manifest))


def infer_pipeline_and_run_id(
    manifest: dict[str, Any], *, manifest_path: Path | None = None
) -> tuple[str, str]:
    """Best-effort pipeline/run_id for attestation ``d`` tags."""
    pipeline = str(
        manifest.get("pipeline") or manifest.get("method") or manifest.get("schema") or "seiso"
    )
    if pipeline in {"lora", "qlora", "full", "slime", "nemo_rl"}:
        pipeline = "training"
    elif "format" in manifest and "file_checksums_sha256" in manifest:
        pipeline = "export"

    run_id = str(manifest.get("run_id") or manifest.get("job_id") or manifest.get("run_name") or "")
    if not run_id and manifest_path is not None:
        # Hash directory names so attestation d-tags do not leak project path labels.
        run_id = hashlib.sha256(manifest_path.parent.name.encode("utf-8")).hexdigest()[:16]
    if not run_id:
        run_id = manifest_sha256_excluding_nostr(manifest)[:16]
    # Keep tag-safe (no whitespace).
    run_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in run_id)[:128]
    pipeline = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in pipeline)[:64]
    return pipeline, run_id


def _chain_or_root_sha256(manifest: dict[str, Any]) -> str | None:
    for key in (
        "chain_head_sha256",
        "integrity_sha256",
        "pipeline_fingerprint",
        "config_sha256",
    ):
        value = manifest.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict) and value:
            return content_fingerprint(value)
    stages = manifest.get("stages")
    if isinstance(stages, list) and stages:
        digests = [str(s.get("sha256")) for s in stages if isinstance(s, dict) and s.get("sha256")]
        if digests:
            return content_fingerprint(digests)
    checksums = manifest.get("file_checksums_sha256")
    if isinstance(checksums, dict) and checksums:
        return content_fingerprint(checksums)
    return None


def build_attestation_v1(
    manifest: dict[str, Any],
    *,
    manifest_path: Path | None = None,
    seiso_version: str | None = None,
) -> dict[str, Any]:
    """Build a digests-only Nostr attestation payload from a local manifest.

    When the manifest includes dataset merkle fields, the schema is
    ``seiso.provenance.attestation/v2`` and those digests are sealed too.
    """
    from seiso import __version__ as pkg_version

    pipeline, run_id = infer_pipeline_and_run_id(manifest, manifest_path=manifest_path)
    body = strip_nostr_receipt(manifest)
    merkle_root = str(manifest.get("dataset_merkle_root") or "").strip() or None
    merkle_count = manifest.get("dataset_merkle_leaf_count")
    merkle_alg = str(manifest.get("dataset_merkle_alg") or "").strip() or None
    schema = ATTESTATION_SCHEMA_V2 if merkle_root else ATTESTATION_SCHEMA_V1
    attestation: dict[str, Any] = {
        "schema": schema,
        "pipeline": pipeline,
        "run_id": run_id,
        "manifest_sha256": content_fingerprint(body),
        "config_fingerprint": (
            str(manifest.get("config_fingerprint") or manifest.get("config_sha256") or "") or None
        ),
        "chain_or_root_sha256": _chain_or_root_sha256(manifest),
        "dataset_merkle_root": merkle_root,
        "dataset_merkle_leaf_count": (
            int(merkle_count) if merkle_root and merkle_count is not None else None
        ),
        "dataset_merkle_alg": merkle_alg if merkle_root else None,
        "git_commit": manifest.get("git_commit"),
        "created_at": manifest.get("created_at")
        or manifest.get("exported_at")
        or manifest.get("completed_at"),
        "seiso_version": seiso_version or pkg_version,
    }
    # Drop nulls for stable compact content.
    return {k: v for k, v in attestation.items() if v is not None and v != ""}


# Alias used by attest / CLI; v1 builder already emits v2 when merkle is present.
build_attestation = build_attestation_v1


def attestation_content_json(attestation: dict[str, Any]) -> str:
    return json.dumps(
        _jsonable(attestation),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def merge_nostr_receipt(manifest: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    """Return manifest copy with ``nostr`` receipt merged in."""
    out = dict(manifest)
    out["nostr"] = dict(receipt)
    return out


def write_manifest_with_nostr_receipt(path: Path, manifest: dict[str, Any]) -> None:
    write_json(path, manifest)
