"""JSONL worker entrypoint for the Rust control plane.

Reads protocol messages from stdin; writes events to stdout (one JSON object
per line). Phase-1 smoke paths run without GPU / PEFT.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1


def emit(msg: dict[str, Any]) -> None:
    msg.setdefault("v", PROTOCOL_VERSION)
    sys.stdout.write(json.dumps(msg, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def handle_train_start(msg: dict[str, Any]) -> None:
    job_id = msg.get("job_id") or "unknown"
    emit(
        {
            "op": "log",
            "job_id": job_id,
            "level": "info",
            "msg": "seiso_ml_worker: train smoke start",
        }
    )
    emit({"op": "progress", "job_id": job_id, "pct": 0.1})
    config = msg.get("config") or {}
    if config.get("smoke_only", True):
        time.sleep(0.05)
        emit({"op": "metric", "job_id": job_id, "name": "loss", "value": 0.42, "step": 1})
        emit({"op": "progress", "job_id": job_id, "pct": 1.0})
        emit(
            {
                "op": "log",
                "job_id": job_id,
                "level": "info",
                "msg": "smoke train complete",
            }
        )
        emit({"op": "done", "job_id": job_id, "status": "ok", "artifacts": []})
        return

    emit(
        {
            "op": "log",
            "job_id": job_id,
            "level": "info",
            "msg": "full train not wired; set smoke_only true",
        }
    )
    emit({"op": "done", "job_id": job_id, "status": "ok", "artifacts": []})


def handle_export_start(msg: dict[str, Any]) -> None:
    job_id = msg.get("job_id") or "unknown"
    paths = msg.get("paths") or {}
    data_dir = Path(paths.get("data_dir") or ".")
    out_dir = data_dir / "exports" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "model.gguf.smoke"
    artifact.write_text("smoke-export\n", encoding="utf-8")
    emit(
        {
            "op": "log",
            "job_id": job_id,
            "level": "info",
            "msg": f"seiso_ml_worker: export smoke -> {artifact}",
        }
    )
    emit({"op": "progress", "job_id": job_id, "pct": 1.0})
    emit(
        {
            "op": "done",
            "job_id": job_id,
            "status": "ok",
            "artifacts": [str(artifact)],
        }
    )


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            emit({"op": "error", "job_id": None, "code": "bad_json", "msg": str(exc)})
            continue
        if msg.get("v") != PROTOCOL_VERSION:
            emit(
                {
                    "op": "error",
                    "job_id": msg.get("job_id"),
                    "code": "bad_version",
                    "msg": f"unsupported v={msg.get('v')}",
                }
            )
            continue
        op = msg.get("op")
        if op == "train.start":
            handle_train_start(msg)
        elif op == "export.start":
            handle_export_start(msg)
        elif op == "cancel":
            emit(
                {
                    "op": "log",
                    "job_id": msg.get("job_id"),
                    "level": "info",
                    "msg": "cancel received",
                }
            )
            emit(
                {
                    "op": "done",
                    "job_id": msg.get("job_id"),
                    "status": "cancelled",
                    "artifacts": [],
                }
            )
        else:
            emit(
                {
                    "op": "error",
                    "job_id": msg.get("job_id"),
                    "code": "unknown_op",
                    "msg": f"unknown op {op!r}",
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
