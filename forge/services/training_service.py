"""Training route helpers: dataset analysis cache and job formatting."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.orchestrators.training import TrainingOrchestrator
from seiso.training.config import DatasetFormat
from seiso.training.dataset_analysis import analyze_training_dataset

_DATASET_ANALYSIS_TTL_S = 10 * 60.0
_CLOUD_GPU_PROVIDER_TYPE = "cloud_gpu"
_CLOUD_SECRET_FIELDS = frozenset(
    {
        "api_key",
        "access_key_id",
        "secret_access_key",
        "session_token",
        "ssh_private_key",
        "bootstrap_command",
    }
)


@dataclass(frozen=True)
class DatasetAnalysisCacheEntry:
    user_id: str
    dataset: str
    requested_format: str
    resolved_format: str
    valid: bool
    created_at: float


_dataset_analysis_cache: dict[str, DatasetAnalysisCacheEntry] = {}


def cloud_gpu_provider_type() -> str:
    return _CLOUD_GPU_PROVIDER_TYPE


def mask_cloud_credential_config(config: dict[str, Any]) -> dict[str, Any]:
    masked: dict[str, Any] = {}
    for key, value in config.items():
        if key in _CLOUD_SECRET_FIELDS:
            masked[f"{key}_configured"] = bool(value)
            continue
        masked[key] = value
    return masked


def cloud_credential_response(row: dict[str, Any]) -> dict[str, Any]:
    import json

    config = json.loads(row["config_json"]) if "config_json" in row else row["config"]
    return {
        "id": row["id"],
        "name": row["name"],
        "provider_type": row["provider_type"],
        "created_at": row["created_at"],
        "config": mask_cloud_credential_config(config),
    }


def store_dataset_analysis_token(
    *,
    user_id: str,
    dataset: str | Path,
    requested_format: DatasetFormat,
    resolved_format: str | None,
    valid: bool,
) -> str:
    token = uuid.uuid4().hex
    _dataset_analysis_cache[token] = DatasetAnalysisCacheEntry(
        user_id=user_id,
        dataset=str(dataset),
        requested_format=requested_format.value,
        resolved_format=resolved_format or requested_format.value,
        valid=valid,
        created_at=time.monotonic(),
    )
    return token


def dataset_analysis_token_matches(
    token: str | None,
    *,
    user_id: str,
    dataset: str | Path,
    dataset_format: DatasetFormat,
) -> bool:
    if not token:
        return False
    entry = _dataset_analysis_cache.get(token)
    now = time.monotonic()
    if entry is None:
        return False
    if now - entry.created_at > _DATASET_ANALYSIS_TTL_S:
        _dataset_analysis_cache.pop(token, None)
        return False
    return (
        entry.user_id == user_id
        and entry.dataset == str(dataset)
        and dataset_format.value in {entry.requested_format, entry.resolved_format}
        and entry.valid
    )


def run_dataset_analysis(
    dataset: str | Path,
    *,
    dataset_format: DatasetFormat,
    sandbox_root: Path,
) -> dict[str, Any]:
    return analyze_training_dataset(
        dataset,
        dataset_format=dataset_format,
        sandbox_root=sandbox_root,
    )


def serialize_metrics_payload(
    points: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from seiso.security.hardware_privacy import sanitize_system_metric_point

    training = [p for p in points if p.get("type") in ("training", "eval")]
    system = [
        sanitize_system_metric_point(p) for p in points if p.get("type") == "system"
    ]
    return {
        "summary": summary or {},
        "training": training[-2000:],
        "system": system[-500:],
        "updated_at": summary.get("updated_at") if summary else None,
    }


def effective_job_status(
    db_status: str,
    orchestrator: TrainingOrchestrator,
    job_id: str,
) -> str:
    live = orchestrator.get_job(job_id)
    if not live:
        return db_status
    live_status = live.status.value
    if db_status == "pending" and live_status in (
        "running",
        "completed",
        "failed",
        "cancelled",
    ):
        return live_status
    return db_status


def format_training_job_row(
    row: dict[str, Any],
    orchestrator: TrainingOrchestrator | None = None,
) -> dict[str, Any]:
    job_id = str(row.get("id", ""))
    status = str(row.get("status", "unknown"))
    if orchestrator is not None:
        status = effective_job_status(status, orchestrator, job_id)
    return {
        "id": job_id,
        "status": status,
        "config_json": row.get("config_json"),
        "metrics_json": row.get("metrics_json"),
        "error_text": row.get("error_text"),
        "checkpoint_path": row.get("checkpoint_path"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "project_id": row.get("project_id"),
    }
