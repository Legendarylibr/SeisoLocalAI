#!/usr/bin/env python3
"""Live Forge integration check — mirrors forge-ui API client routes."""

from __future__ import annotations

import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8765"
API = f"{BASE}/api"
REPO = Path(__file__).resolve().parents[1]


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Report:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append(CheckResult(name, ok, detail))
        mark = "OK" if ok else "FAIL"
        line = f"[{mark}] {name}"
        if detail:
            line += f" — {detail}"
        print(line)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.ok)


def mint_token() -> tuple[str, str]:
    from forge.config import ForgeSettings
    from forge.security.auth import create_access_token

    settings = ForgeSettings()
    db_path = settings.data_dir / "forge.db"
    if not db_path.is_file():
        raise RuntimeError(f"forge.db not found at {db_path}")
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT id FROM users LIMIT 1").fetchone()
    conn.close()
    if not row:
        raise RuntimeError("no users in forge.db — complete onboarding first")
    user_id = row[0]
    token = create_access_token(user_id, settings)
    return user_id, token


def check_static_ui(client: httpx.Client, report: Report) -> None:
    for path in (
        "/",
        "/train",
        "/chat",
        "/export",
        "/compress",
        "/distill-rl",
    ):
        r = client.get(f"{BASE}{path}")
        report.add(
            f"SPA route {path}",
            r.status_code == 200 and "text/html" in r.headers.get("content-type", ""),
            f"status={r.status_code}",
        )
    assets = REPO / "forge-ui" / "dist" / "assets"
    js_files = list(assets.glob("TrainPage-*.js"))
    report.add(
        "TrainPage bundle built",
        bool(js_files),
        str(js_files[0].name if js_files else "missing"),
    )


def api_get(client: httpx.Client, path: str, headers: dict[str, str]) -> httpx.Response:
    return client.get(f"{API}{path}", headers=headers)


def api_post(
    client: httpx.Client, path: str, headers: dict[str, str], body: dict
) -> httpx.Response:
    return client.post(f"{API}{path}", headers=headers, json=body)


def check_training_flow(
    client: httpx.Client, headers: dict[str, str], report: Report
) -> str | None:
    r = api_get(client, "/training/models", headers)
    report.add("GET /training/models", r.status_code == 200, f"status={r.status_code}")
    models = r.json().get("models", []) if r.status_code == 200 else []

    r = api_get(client, "/training/datasets?q=alpaca&limit=5", headers)
    report.add(
        "GET /training/datasets",
        r.status_code == 200,
        f"total={r.json().get('total') if r.is_success else r.status_code}",
    )

    model_id = ""
    if models:
        model_id = models[0].get("repo_id") or models[0].get("model_id") or ""
    if not model_id:
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"

    r = api_get(
        client,
        f"/training/recommendations?model_id={model_id}&dataset=./data/sample.jsonl",
        headers,
    )
    report.add(
        "GET /training/recommendations", r.status_code == 200, f"status={r.status_code}"
    )

    r = api_post(
        client,
        "/training/analyze-dataset",
        headers,
        {"dataset": "./data/sample.jsonl", "dataset_format": "auto"},
    )
    analysis = r.json() if r.status_code == 200 else {}
    report.add(
        "POST /training/analyze-dataset",
        r.status_code == 200 and analysis.get("valid", False),
        f"rows={analysis.get('row_count')} format={analysis.get('detected_format')}",
    )

    r = api_post(
        client,
        "/training/validate-dataset",
        headers,
        {"dataset": "./data/sample.jsonl", "dataset_format": "auto"},
    )
    valid = r.json() if r.status_code == 200 else {}
    report.add(
        "POST /training/validate-dataset",
        r.status_code == 200 and valid.get("valid", False),
        valid.get("error") or "valid",
    )

    r = api_get(client, "/training/jobs", headers)
    report.add(
        "GET /training/jobs",
        r.status_code == 200,
        f"count={len(r.json()) if r.is_success else 0}",
    )

    cfg = {
        "model_id": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "dataset": "./data/sample.jsonl",
        "output_dir": f"./.test_outputs/live-ui-{uuid.uuid4().hex[:8]}",
        "method": "lora",
        "quant": "16bit",
        "dataset_format": "chat",
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 0.0002,
        "max_seq_length": 128,
        "lora_r": 4,
        "lora_alpha": 8,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "train_on_responses_only": True,
        "use_triton": False,
        "use_fused_ce": False,
        "use_fused_lora": False,
        "eval_split_ratio": 0,
        "save_steps": 10,
        "logging_steps": 1,
        "seed": 42,
        "deterministic": True,
    }
    r = api_post(client, "/training/jobs", headers, {"config": cfg, "multi_gpu": False})
    job_id = None
    if r.status_code == 200:
        job_id = r.json().get("job_id")
    report.add("POST /training/jobs (smoke)", r.status_code == 200, f"job_id={job_id}")

    if job_id:
        deadline = time.time() + 180
        terminal = {"completed", "failed", "cancelled"}
        status = "pending"
        while time.time() < deadline:
            rows = api_get(client, "/training/jobs", headers).json()
            row = next((j for j in rows if j.get("id") == job_id), None)
            status = row.get("status", status) if row else status
            if status in terminal:
                break
            time.sleep(2)
        report.add(
            "Training job reaches terminal state",
            status in terminal,
            f"status={status} error={row.get('error_text') if row else None}",
        )

        r = api_get(client, f"/training/jobs/{job_id}/metrics", headers)
        report.add(
            "GET /training/jobs/{id}/metrics",
            r.status_code == 200,
            f"status={r.status_code}",
        )

        # SSE smoke — read first event
        with client.stream(
            "GET",
            f"{API}/training/jobs/{job_id}/stream",
            headers=headers,
            timeout=10.0,
        ) as stream:
            got_event = False
            for line in stream.iter_lines():
                if line.startswith("data:"):
                    got_event = True
                    break
        report.add(
            "GET /training/jobs/{id}/stream (SSE)",
            got_event or status in terminal,
            "event or terminal",
        )

    return job_id


def check_other_pages(
    client: httpx.Client, headers: dict[str, str], report: Report
) -> None:
    endpoints: list[tuple[str, str]] = [
        ("GET /auth/status", "GET", "/auth/status"),
        ("GET /auth/me", "GET", "/auth/me"),
        ("GET /settings", "GET", "/settings"),
        ("GET /settings/hf-status", "GET", "/settings/hf-status"),
        ("GET /system/hardware", "GET", "/system/hardware"),
        ("GET /system/metrics", "GET", "/system/metrics"),
        ("GET /models", "GET", "/models"),
        ("GET /models/vram", "GET", "/models/vram"),
        ("GET /models/catalog", "GET", "/models/catalog?limit=5&purpose=chat"),
        ("GET /export/profiles", "GET", "/export/profiles"),
        ("GET /export/jobs", "GET", "/export/jobs"),
        ("GET /export/publishable", "GET", "/export/publishable"),
        ("GET /compress/presets", "GET", "/compress/presets"),
        ("GET /compress/jobs", "GET", "/compress/jobs"),
        ("GET /distill-rl/presets", "GET", "/distill-rl/presets"),
        ("GET /distill-rl/jobs", "GET", "/distill-rl/jobs"),
        ("POST /recipes/jobs (empty recipe)", "POST", "/recipes/jobs"),
        ("GET /knowledge/bases", "GET", "/knowledge/bases"),
        ("GET /providers", "GET", "/providers"),
    ]
    for label, method, path in endpoints:
        if method == "GET":
            r = api_get(client, path, headers)
            ok = r.status_code == 200
        elif path == "/recipes/jobs":
            r = api_post(client, path, headers, {"recipe": {"nodes": [], "edges": []}})
            ok = r.status_code == 200
        else:
            r = api_post(client, path, headers, {})
            ok = r.status_code == 200
        report.add(label, ok, f"status={r.status_code}")


def check_compat_route(client: httpx.Client, report: Report) -> None:
    key_path = Path.home() / ".seiso" / ".inference_api_key"
    if not key_path.is_file():
        report.add("Compat /v1/models", False, "missing inference API key")
        return
    key = key_path.read_text(encoding="utf-8").strip()
    r = client.get(
        f"{BASE}/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    report.add("GET /v1/models", r.status_code == 200, f"status={r.status_code}")


def main() -> int:
    report = Report()
    try:
        user_id, token = mint_token()
        headers = {"Authorization": f"Bearer {token}"}
        report.add("Mint session token", True, f"user={user_id[:8]}…")
    except Exception as exc:
        report.add("Mint session token", False, str(exc))
        return 1

    with httpx.Client(timeout=30.0) as client:
        r = client.get(f"{BASE}/health")
        report.add(
            "GET /health",
            r.status_code == 200 and r.json().get("status") == "ok",
            str(r.json()),
        )

        check_static_ui(client, report)
        check_other_pages(client, headers, report)
        check_training_flow(client, headers, report)
        check_compat_route(client, report)

    print()
    print(f"Summary: {report.passed} passed, {report.failed} failed")
    if report.failed:
        print("Failures:")
        for item in report.results:
            if not item.ok:
                print(f"  - {item.name}: {item.detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
