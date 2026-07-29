"""Marketplace job runner — wraps existing Seiso CLI trainers in a pay sandbox."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from seiso.pay.ark import settle_split
from seiso.pay.flags import require_pay_allowed
from seiso.pay.pricing import quote_job
from seiso.pay.store import (
    create_job,
    escrow_hold,
    escrow_release_refund,
    load_job,
    pay_root,
    save_job,
)
from seiso.security import SecurityError, assert_within, resolve_data_dir, safe_join

_JOB_LOCK = threading.Lock()
_RUNNING = False

# Allowlisted job types → CLI argv builder
_RL_TYPES = frozenset({"slime", "distill_rl", "rl_quant", "nemo_rl"})


def _append_log(job: dict[str, Any], line: str, data_dir: Path | None) -> None:
    tail = list(job.get("log_tail") or [])
    tail.append(line.rstrip()[:2000])
    job["log_tail"] = tail[-200:]
    save_job(job, data_dir)


def _already_refunded(job: dict[str, Any]) -> bool:
    settlement = job.get("settlement") or {}
    if settlement.get("status") == "refunded":
        return True
    return int(job.get("refunded_sats") or 0) > 0


def _reload_job(job: dict[str, Any], data_dir: Path | None) -> dict[str, Any]:
    """Refresh job from disk (cancel may have raced the runner)."""
    try:
        return load_job(str(job["job_id"]), data_dir)
    except KeyError:
        return job


def _is_cancelled_or_refunded(job: dict[str, Any]) -> bool:
    if job.get("status") == "cancelled":
        return True
    return _already_refunded(job)


def _refund_escrow(
    job: dict[str, Any],
    *,
    data_dir: Path | None,
    reason: str,
) -> int:
    """Return escrowed sats to session balance; idempotent per job.

    Session-level ``escrow_release_refund(..., job_id=)`` is the durable
    anti-double-credit gate. Job markers are written after the credit so a
    crash mid-refund retries safely (session no-ops; markers get filled in).

    Mutates ``job`` in place (does not replace the dict) so callers that set
    ``status``/``error`` before refunding do not lose those fields on save.
    """
    fresh = _reload_job(job, data_dir)
    if _already_refunded(fresh):
        job["refunded_sats"] = int(fresh.get("refunded_sats") or 0)
        if fresh.get("settlement"):
            job["settlement"] = fresh["settlement"]
        return int(job.get("refunded_sats") or 0)
    total = int((job.get("quote") or fresh.get("quote") or {}).get("total_sats") or 0)
    if total <= 0:
        return 0
    escrow_release_refund(
        str(job["session_id"]),
        amount_sats=total,
        data_dir=data_dir,
        job_id=str(job["job_id"]),
        reason=reason,
    )
    job["refunded_sats"] = total
    job["settlement"] = {
        "status": "refunded",
        "amount_sats": total,
        "reason": reason,
        "rail": "session_balance",
        "detail": (
            "Escrow restored to prepaid session balance. "
            "Lightning/L402 pay-in is one-way; no on-chain reverse payment."
        ),
        "ts": time.time(),
    }
    save_job(job, data_dir)
    return total


def _sandbox_config_path(config: str | None) -> str | None:
    """Restrict buyer ``-c`` paths to ``configs/`` under the process cwd."""
    if not config:
        return None
    raw = str(config).strip()
    if not raw:
        return None
    configs_root = (Path.cwd() / "configs").resolve()
    candidate = Path(raw)
    if not candidate.is_absolute():
        # Allow both ``configs/foo.yaml`` and ``foo.yaml`` (under configs/).
        parts = candidate.parts
        if parts and parts[0] == "configs":
            candidate = Path.cwd() / candidate
        else:
            try:
                candidate = safe_join(configs_root, *parts)
            except SecurityError as exc:
                raise ValueError(
                    "config must be a relative path under configs/ "
                    f"(rejected: {raw!r})"
                ) from exc
            return str(candidate)
    try:
        resolved = assert_within(configs_root, candidate)
    except SecurityError as exc:
        raise ValueError(
            "config must resolve under configs/ " f"(rejected: {raw!r})"
        ) from exc
    if not resolved.is_file():
        raise ValueError(f"config not found under configs/: {raw}")
    return str(resolved)


def start_job(
    *,
    session_id: str,
    job_type: str,
    preset: str | None = None,
    config: str | None = None,
    data_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Quote, escrow, and run (or dry-run) a marketplace job."""
    require_pay_allowed()
    jt = job_type.strip().lower()
    quote = quote_job(jt, preset=preset)
    sandboxed = _sandbox_config_path(config)
    job = create_job(
        session_id=session_id,
        job_type=jt,
        quote=quote,
        preset=preset,
        config_path=sandboxed,
        data_dir=data_dir,
    )
    total = int(quote["total_sats"])
    escrow_hold(
        session_id,
        total_sats=total,
        data_dir=data_dir,
        job_id=job["job_id"],
    )
    job["status"] = "running"
    save_job(job, data_dir)

    if dry_run:
        return _complete_job(job, data_dir=data_dir, dry_run=True)

    global _RUNNING
    with _JOB_LOCK:
        if _RUNNING:
            job["status"] = "failed"
            job["error"] = "another marketplace GPU job is already running"
            _refund_escrow(job, data_dir=data_dir, reason="gpu_busy")
            save_job(job, data_dir)
            return job
        _RUNNING = True

    try:
        return _execute_job(job, data_dir=data_dir)
    finally:
        with _JOB_LOCK:
            _RUNNING = False


def _artifact_dir(job: dict[str, Any], data_dir: Path | None) -> Path:
    root = pay_root(data_dir)
    path = safe_join(
        root, "artifacts", str(job["session_id"]), str(job["job_id"])
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cli_command(job: dict[str, Any], artifact: Path) -> list[str]:
    jt = job["job_type"]
    preset = (job.get("preset") or "smoke").strip().lower()
    config = job.get("config_path")
    py = sys.executable

    if jt == "finetune":
        cfg = config or "configs/smoke_train_cpu.yaml"
        return [py, "-m", "seiso_cli.main", "train", "-c", cfg]

    if jt == "slime":
        cfg = config or "configs/smoke_train_cpu.yaml"
        return [py, "-m", "seiso_cli.main", "slime", "-c", cfg]

    if jt == "distill_rl":
        args = [py, "-m", "seiso_cli.main", "distill-rl", "run", "--preset", preset]
        return args

    if jt == "rl_quant":
        p = "minimal" if preset in {"smoke", "minimal", ""} else preset
        return [py, "-m", "seiso_cli.main", "rl-quant", "run", "--preset", p]

    if jt == "nemo_rl":
        if not (os.environ.get("SEISO_NEMO_RL_ROOT") or "").strip():
            raise RuntimeError(
                "nemo_rl requires SEISO_NEMO_RL_ROOT on the operator host"
            )
        cfg = config or "configs/smoke_nemo_rl.yaml"
        return [py, "-m", "seiso_cli.main", "nemo-rl", "-c", cfg]

    raise ValueError(f"unsupported job type: {jt}")


def _execute_job(job: dict[str, Any], *, data_dir: Path | None) -> dict[str, Any]:
    artifact = _artifact_dir(job, data_dir)
    job["artifact_dir"] = str(artifact)
    save_job(job, data_dir)

    fresh = _reload_job(job, data_dir)
    if _is_cancelled_or_refunded(fresh):
        return fresh

    try:
        cmd = _cli_command(job, artifact)
    except Exception as exc:
        return _fail_job(job, str(exc), data_dir=data_dir)

    _append_log(job, f"$ {' '.join(cmd)}", data_dir)
    env = os.environ.copy()
    # Isolate marketplace artifacts under pay sandbox when possible
    env.setdefault("SEISO_DATA_DIR", str(resolve_data_dir(data_dir)))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path.cwd()),
            check=False,
            timeout=int(os.environ.get("SEISO_PAY_JOB_TIMEOUT_S") or 3600),
        )
    except subprocess.TimeoutExpired:
        return _fail_job(job, "job wall-clock timeout", data_dir=data_dir)
    except FileNotFoundError as exc:
        return _fail_job(job, f"runner missing: {exc}", data_dir=data_dir)

    fresh = _reload_job(job, data_dir)
    if _is_cancelled_or_refunded(fresh):
        # Cancel won the race — do not settle as paid after refund.
        if proc.stdout:
            for line in proc.stdout.splitlines()[-20:]:
                _append_log(fresh, line, data_dir)
        return fresh

    if proc.stdout:
        for line in proc.stdout.splitlines()[-50:]:
            _append_log(job, line, data_dir)
    if proc.stderr:
        for line in proc.stderr.splitlines()[-50:]:
            _append_log(job, line, data_dir)

    if proc.returncode != 0:
        return _fail_job(
            job,
            f"exit {proc.returncode}",
            data_dir=data_dir,
            refund=True,
        )

    manifest = {
        "job_id": job["job_id"],
        "job_type": job["job_type"],
        "artifact_dir": str(artifact),
        "quote": job.get("quote"),
    }
    (artifact / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return _complete_job(job, data_dir=data_dir, dry_run=False)


def _complete_job(
    job: dict[str, Any], *, data_dir: Path | None, dry_run: bool
) -> dict[str, Any]:
    fresh = _reload_job(job, data_dir)
    if _is_cancelled_or_refunded(fresh):
        return fresh

    quote = job["quote"]
    try:
        receipt = settle_split(
            compute_sats=int(quote["compute_sats"]),
            protocol_fee_sats=int(quote["protocol_fee_sats"]),
            job_id=job["job_id"],
            session_id=job["session_id"],
        )
    except Exception as exc:
        return _fail_job(
            job,
            f"settle failed: {exc}",
            data_dir=data_dir,
            refund=True,
        )

    # Re-check cancel after settle_split (slow path) before recording spend.
    fresh = _reload_job(job, data_dir)
    if _is_cancelled_or_refunded(fresh):
        return fresh

    # Escrow already removed total from balance; record spend split on session.
    from seiso.pay.store import append_ledger, record_session_spend

    record_session_spend(
        job["session_id"],
        compute_sats=int(quote["compute_sats"]),
        protocol_fee_sats=int(quote["protocol_fee_sats"]),
        data_dir=data_dir,
    )
    append_ledger(
        {
            "event": "job_settle",
            "job_id": job["job_id"],
            "session_id": job["session_id"],
            "settlement": receipt.as_dict(),
            "dry_run": dry_run,
            "ts": time.time(),
        },
        data_dir=data_dir,
    )
    job["status"] = "completed"
    job["settlement"] = receipt.as_dict()
    # Do not clear refunded_sats — a cancelled/refunded job must keep markers.
    if dry_run:
        job["artifact_dir"] = job.get("artifact_dir") or str(
            _artifact_dir(job, data_dir)
        )
        _append_log(job, "dry_run complete (no trainer invoked)", data_dir)
    save_job(job, data_dir)
    return job


def _fail_job(
    job: dict[str, Any],
    error: str,
    *,
    data_dir: Path | None,
    refund: bool = True,
) -> dict[str, Any]:
    fresh = _reload_job(job, data_dir)
    if fresh.get("status") == "cancelled" or _already_refunded(fresh):
        # Preserve cancel/refund outcome; attach error note if useful.
        if error and not fresh.get("error"):
            fresh["error"] = error
            save_job(fresh, data_dir)
        return fresh
    job["status"] = "failed"
    job["error"] = error
    _append_log(job, f"ERROR: {error}", data_dir)
    if refund:
        _refund_escrow(job, data_dir=data_dir, reason="job_failure")
    save_job(job, data_dir)
    return job


def cancel_job(job_id: str, *, data_dir: Path | None = None) -> dict[str, Any]:
    require_pay_allowed()
    job = load_job(job_id, data_dir)
    if job.get("status") in {"completed", "failed", "cancelled"}:
        return job
    prev = job.get("status")
    job["status"] = "cancelled"
    job["error"] = "cancelled by client"
    save_job(job, data_dir)  # durable cancel flag before refund (runner races)
    if prev in {"pending", "running"}:
        _refund_escrow(job, data_dir=data_dir, reason="cancelled")
    return load_job(job_id, data_dir)


def job_receipt(job: dict[str, Any]) -> dict[str, Any]:
    q = job.get("quote") or {}
    settlement = job.get("settlement")
    refunded = int(job.get("refunded_sats") or 0)
    return {
        "mode": "paid",
        "type": job.get("job_type"),
        "status": job.get("status"),
        "job_id": job.get("job_id"),
        "compute_sats": q.get("compute_sats"),
        "protocol_fee_sats": q.get("protocol_fee_sats"),
        "total_sats": q.get("total_sats"),
        "refunded_sats": refunded,
        "artifacts": job.get("artifact_dir"),
        "settlement": settlement,
        "error": job.get("error"),
    }
