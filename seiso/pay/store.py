"""File-backed session / job / ledger store under ``$SEISO_DATA_DIR/pay/``."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
import uuid
from pathlib import Path
from typing import Any, cast

from seiso.security import resolve_data_dir, safe_join

_TOKEN_PREFIX = "seiso_pay_"  # nosec B105 — public token namespace prefix, not a secret
_lock = threading.RLock()


def pay_root(data_dir: Path | None = None) -> Path:
    root = resolve_data_dir(data_dir)
    path = safe_join(root, "pay")
    path.mkdir(parents=True, exist_ok=True)
    for sub in ("sessions", "jobs", "ledger", "artifacts"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_token() -> str:
    return _TOKEN_PREFIX + secrets.token_urlsafe(32)


def create_session(
    *,
    scopes: list[str],
    data_dir: Path | None = None,
    funding_mode: str = "pending",
) -> dict[str, Any]:
    token = new_token()
    session_id = uuid.uuid4().hex
    now = time.time()
    record: dict[str, Any] = {
        "session_id": session_id,
        "token_hash": _hash_token(token),
        "token_prefix": token[:16] + "…",
        "scopes": sorted({s.strip().lower() for s in scopes if s.strip()}),
        "status": "pending",
        "balance_sats": 0,
        "funded_sats": 0,
        "spent_compute_sats": 0,
        "spent_protocol_fee_sats": 0,
        "refunded_sats": 0,
        "funding_mode": funding_mode,
        "created_at": now,
        "updated_at": now,
    }
    path = safe_join(pay_root(data_dir), "sessions", f"{session_id}.json")
    with _lock:
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        _append_ledger(
            {
                "event": "session_create",
                "session_id": session_id,
                "scopes": record["scopes"],
                "ts": now,
            },
            data_dir=data_dir,
        )
    # Return plaintext token once (caller stores in env); never re-read from disk.
    out = dict(record)
    out["token"] = token
    return out


def _session_path(session_id: str, data_dir: Path | None = None) -> Path:
    return safe_join(pay_root(data_dir), "sessions", f"{session_id}.json")


def load_session(session_id: str, data_dir: Path | None = None) -> dict[str, Any]:
    path = _session_path(session_id, data_dir)
    if not path.is_file():
        raise KeyError(f"session not found: {session_id}")
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _save_session(record: dict[str, Any], data_dir: Path | None = None) -> None:
    record["updated_at"] = time.time()
    path = _session_path(str(record["session_id"]), data_dir)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def resolve_session_by_token(token: str, data_dir: Path | None = None) -> dict[str, Any]:
    token = (token or "").strip()
    if not token.startswith(_TOKEN_PREFIX):
        raise KeyError("invalid pay token")
    digest = _hash_token(token)
    root = pay_root(data_dir) / "sessions"
    with _lock:
        for path in root.glob("*.json"):
            record = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
            if record.get("token_hash") == digest:
                return record
    raise KeyError("invalid pay token")


def activate_session(
    session_id: str,
    *,
    amount_sats: int,
    data_dir: Path | None = None,
    funding_mode: str = "faucet",
    fund_id: str | None = None,
) -> dict[str, Any]:
    """Credit session balance.

    When ``fund_id`` is set (e.g. L402 ``challenge_id``), the credit is
    idempotent: a second call with the same id is a no-op that returns the
    current session. Persist the fund id on the session *with* the balance
    bump so crash/retry cannot double-credit.
    """
    if amount_sats <= 0:
        raise ValueError("amount_sats must be > 0")
    with _lock:
        record = load_session(session_id, data_dir)
        fund_ids = record.setdefault("fund_ids", [])
        if not isinstance(fund_ids, list):
            fund_ids = []
            record["fund_ids"] = fund_ids
        if fund_id and fund_id in fund_ids:
            return record
        record["balance_sats"] = int(record.get("balance_sats") or 0) + amount_sats
        record["funded_sats"] = int(record.get("funded_sats") or 0) + amount_sats
        record["status"] = "active"
        record["funding_mode"] = funding_mode
        if fund_id:
            fund_ids.append(fund_id)
        _save_session(record, data_dir)
        _append_ledger(
            {
                "event": "session_fund",
                "session_id": session_id,
                "amount_sats": amount_sats,
                "funding_mode": funding_mode,
                "fund_id": fund_id,
                "balance_sats": record["balance_sats"],
                "ts": time.time(),
            },
            data_dir=data_dir,
        )
    return record


def debit_session(
    session_id: str,
    *,
    compute_sats: int,
    protocol_fee_sats: int,
    data_dir: Path | None = None,
    reason: str = "debit",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total = int(compute_sats) + int(protocol_fee_sats)
    if total <= 0:
        raise ValueError("debit total must be > 0")
    with _lock:
        record = load_session(session_id, data_dir)
        if record.get("status") != "active":
            raise RuntimeError(f"session not active: {record.get('status')}")
        bal = int(record.get("balance_sats") or 0)
        if bal < total:
            raise RuntimeError(f"insufficient balance: need {total} sats, have {bal}")
        record["balance_sats"] = bal - total
        record["spent_compute_sats"] = int(record.get("spent_compute_sats") or 0) + int(
            compute_sats
        )
        record["spent_protocol_fee_sats"] = int(record.get("spent_protocol_fee_sats") or 0) + int(
            protocol_fee_sats
        )
        _save_session(record, data_dir)
        entry = {
            "event": "debit",
            "session_id": session_id,
            "reason": reason,
            "compute_sats": int(compute_sats),
            "protocol_fee_sats": int(protocol_fee_sats),
            "total_sats": total,
            "balance_sats": record["balance_sats"],
            "ts": time.time(),
        }
        if meta:
            entry["meta"] = meta
        _append_ledger(entry, data_dir=data_dir)
    return record


def escrow_hold(
    session_id: str,
    *,
    total_sats: int,
    data_dir: Path | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Reserve sats for a job (debit immediately into escrow accounting)."""
    # For v1 ledger simplicity: debit full total up front; refund on cancel.
    # Split tracked at settle time via job record.
    with _lock:
        record = load_session(session_id, data_dir)
        if record.get("status") != "active":
            raise RuntimeError(f"session not active: {record.get('status')}")
        bal = int(record.get("balance_sats") or 0)
        if bal < total_sats:
            raise RuntimeError(f"insufficient balance for escrow: need {total_sats}, have {bal}")
        record["balance_sats"] = bal - total_sats
        _save_session(record, data_dir)
        _append_ledger(
            {
                "event": "escrow_hold",
                "session_id": session_id,
                "job_id": job_id,
                "total_sats": total_sats,
                "balance_sats": record["balance_sats"],
                "ts": time.time(),
            },
            data_dir=data_dir,
        )
    return record


def escrow_release_refund(
    session_id: str,
    *,
    amount_sats: int,
    data_dir: Path | None = None,
    job_id: str | None = None,
    reason: str = "job_failure",
) -> dict[str, Any]:
    """Credit escrow back to the prepaid session balance.

    Lightning/L402 pay-in is one-way; marketplace refunds restore session
    balance (not an on-chain/LN reverse payment).

    When ``job_id`` is set, the credit is idempotent per job so crash/retry
    or cancel+fail cannot double-refund.
    """
    if amount_sats <= 0:
        return load_session(session_id, data_dir)
    with _lock:
        record = load_session(session_id, data_dir)
        refund_ids = record.setdefault("escrow_refund_ids", {})
        if not isinstance(refund_ids, dict):
            refund_ids = {}
            record["escrow_refund_ids"] = refund_ids
        if job_id and job_id in refund_ids:
            return record
        record["balance_sats"] = int(record.get("balance_sats") or 0) + int(amount_sats)
        record["refunded_sats"] = int(record.get("refunded_sats") or 0) + int(amount_sats)
        if job_id:
            refund_ids[job_id] = int(amount_sats)
        _save_session(record, data_dir)
        _append_ledger(
            {
                "event": "escrow_refund",
                "session_id": session_id,
                "job_id": job_id,
                "amount_sats": amount_sats,
                "reason": reason,
                "balance_sats": record["balance_sats"],
                "ts": time.time(),
            },
            data_dir=data_dir,
        )
    return record


def create_job(
    *,
    session_id: str,
    job_type: str,
    quote: dict[str, Any],
    preset: str | None = None,
    config_path: str | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    now = time.time()
    record: dict[str, Any] = {
        "job_id": job_id,
        "session_id": session_id,
        "job_type": job_type.strip().lower(),
        "preset": preset,
        "config_path": config_path,
        "status": "pending",
        "quote": quote,
        "created_at": now,
        "updated_at": now,
        "log_tail": [],
        "artifact_dir": None,
        "error": None,
        "settlement": None,
        "refunded_sats": 0,
    }
    path = safe_join(pay_root(data_dir), "jobs", f"{job_id}.json")
    with _lock:
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        _append_ledger(
            {
                "event": "job_create",
                "job_id": job_id,
                "session_id": session_id,
                "job_type": record["job_type"],
                "total_sats": quote.get("total_sats"),
                "ts": now,
            },
            data_dir=data_dir,
        )
    return record


def load_job(job_id: str, data_dir: Path | None = None) -> dict[str, Any]:
    path = safe_join(pay_root(data_dir), "jobs", f"{job_id}.json")
    if not path.is_file():
        raise KeyError(f"job not found: {job_id}")
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def save_job(record: dict[str, Any], data_dir: Path | None = None) -> None:
    record["updated_at"] = time.time()
    path = safe_join(pay_root(data_dir), "jobs", f"{record['job_id']}.json")
    with _lock:
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def list_jobs(
    *,
    session_id: str | None = None,
    data_dir: Path | None = None,
) -> list[dict[str, Any]]:
    root = pay_root(data_dir) / "jobs"
    jobs: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        rec = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        if session_id and rec.get("session_id") != session_id:
            continue
        jobs.append(rec)
    return jobs


def append_ledger(entry: dict[str, Any], *, data_dir: Path | None = None) -> None:
    day = time.strftime("%Y%m%d", time.gmtime())
    path = safe_join(pay_root(data_dir), "ledger", f"{day}.jsonl")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, separators=(",", ":")) + "\n")


# Back-compat alias used internally during create/fund paths.
_append_ledger = append_ledger


def record_session_spend(
    session_id: str,
    *,
    compute_sats: int,
    protocol_fee_sats: int,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Accumulate spend counters after escrow was already held."""
    with _lock:
        record = load_session(session_id, data_dir)
        record["spent_compute_sats"] = int(record.get("spent_compute_sats") or 0) + int(
            compute_sats
        )
        record["spent_protocol_fee_sats"] = int(record.get("spent_protocol_fee_sats") or 0) + int(
            protocol_fee_sats
        )
        _save_session(record, data_dir)
    return record


def public_session_view(record: dict[str, Any]) -> dict[str, Any]:
    """Strip secrets for API/CLI status output."""
    return {
        "session_id": record.get("session_id"),
        "status": record.get("status"),
        "scopes": record.get("scopes"),
        "balance_sats": record.get("balance_sats"),
        "funded_sats": record.get("funded_sats"),
        "spent_compute_sats": record.get("spent_compute_sats"),
        "spent_protocol_fee_sats": record.get("spent_protocol_fee_sats"),
        "refunded_sats": int(record.get("refunded_sats") or 0),
        "funding_mode": record.get("funding_mode"),
        "token_prefix": record.get("token_prefix"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
    }
