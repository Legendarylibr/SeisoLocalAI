"""Opt-in sats marketplace — fee split, sessions, faucet, dry-run jobs."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def pay_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SEISO_ALLOW_PAY", "1")
    monkeypatch.setenv("SEISO_PAY_FAUCET", "1")
    monkeypatch.setenv("SEISO_PROTOCOL_FEE_BPS", "500")
    monkeypatch.setenv("SEISO_PROTOCOL_TREASURY_ARK", "ark1testtreasury")
    monkeypatch.setenv("SEISO_OPERATOR_ARK", "ark1testoperator")
    monkeypatch.delenv("SEISO_ARK_BACKEND", raising=False)
    return tmp_path / "data"


def test_pay_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEISO_ALLOW_PAY", raising=False)
    from seiso.pay.flags import pay_allowed, require_pay_allowed

    assert pay_allowed() is False
    with pytest.raises(RuntimeError, match="SEISO_ALLOW_PAY"):
        require_pay_allowed()


def test_fee_split_five_percent(pay_env: Path) -> None:
    from seiso.pay.pricing import fee_split, quote_job

    split = fee_split(10_000)
    assert split.compute_sats == 10_000
    assert split.protocol_fee_bps == 500
    assert split.protocol_fee_sats == 500
    assert split.total_sats == 10_500
    assert split.payee_operator_sats == 10_000
    assert split.payee_protocol_sats == 500

    q = quote_job("finetune", preset="smoke")
    assert q["protocol_fee_sats"] == (q["compute_sats"] * 500 + 9999) // 10_000
    assert q["total_sats"] == q["compute_sats"] + q["protocol_fee_sats"]


def test_fee_bps_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEISO_PROTOCOL_FEE_BPS", "1500")
    monkeypatch.delenv("SEISO_PROTOCOL_FEE_OVERRIDE", raising=False)
    from seiso.pay.flags import protocol_fee_bps

    with pytest.raises(ValueError, match="exceeds max"):
        protocol_fee_bps()


def test_session_faucet_and_dry_run_job(pay_env: Path) -> None:
    from seiso.pay.jobs import job_receipt, start_job
    from seiso.pay.store import activate_session, create_session, load_session

    created = create_session(scopes=["finetune", "rl", "inference"])
    token = created["token"]
    assert token.startswith("seiso_pay_")
    activate_session(created["session_id"], amount_sats=50_000, funding_mode="faucet")
    session = load_session(created["session_id"])
    assert session["status"] == "active"
    assert session["balance_sats"] == 50_000

    job = start_job(
        session_id=created["session_id"],
        job_type="finetune",
        preset="smoke",
        dry_run=True,
    )
    assert job["status"] == "completed"
    receipt = job_receipt(job)
    assert receipt["protocol_fee_sats"] == job["quote"]["protocol_fee_sats"]
    assert job["settlement"]["status"] == "settled"

    session2 = load_session(created["session_id"])
    assert session2["balance_sats"] == 50_000 - int(job["quote"]["total_sats"])
    assert session2["spent_protocol_fee_sats"] == job["quote"]["protocol_fee_sats"]

    # token resolves
    from seiso.pay.store import resolve_session_by_token

    assert resolve_session_by_token(token)["session_id"] == created["session_id"]


def test_rl_dry_run_jobs(pay_env: Path) -> None:
    from seiso.pay.jobs import start_job
    from seiso.pay.store import activate_session, create_session

    created = create_session(scopes=["rl"])
    activate_session(created["session_id"], amount_sats=100_000, funding_mode="faucet")
    for jt, preset in (
        ("slime", "smoke"),
        ("distill_rl", "smoke"),
        ("rl_quant", "minimal"),
    ):
        job = start_job(
            session_id=created["session_id"],
            job_type=jt,
            preset=preset,
            dry_run=True,
        )
        assert job["status"] == "completed", jt
        assert job["quote"]["protocol_fee_sats"] > 0


def test_inference_debit(pay_env: Path) -> None:
    from seiso.pay.inference import debit_inference
    from seiso.pay.store import activate_session, create_session, load_session

    created = create_session(scopes=["inference"])
    activate_session(created["session_id"], amount_sats=1_000, funding_mode="faucet")
    before = load_session(created["session_id"])["balance_sats"]
    meter = debit_inference(
        created["session_id"],
        prompt_tokens=100,
        completion_tokens=50,
    )
    assert "quote" in meter
    assert meter["quote"]["protocol_fee_sats"] >= 1
    after = load_session(created["session_id"])["balance_sats"]
    assert after == before - meter["quote"]["total_sats"]


def test_well_known_and_app_health(pay_env: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from seiso.pay.app import build_app

    client = TestClient(build_app())
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["pay_allowed"] is True
    wk = client.get("/.well-known/seiso-pay.json")
    assert wk.status_code == 200
    assert wk.json()["protocol_fee_bps"] == 500

    created = client.post(
        "/pay/v1/sessions",
        json={"scopes": ["finetune"], "sats": 20_000},
    )
    assert created.status_code == 200
    body = created.json()
    assert body["token"].startswith("seiso_pay_")
    assert body["session"]["status"] == "active"

    q = client.post("/pay/v1/quotes", json={"type": "finetune", "preset": "smoke"})
    assert q.status_code == 200
    assert "protocol_fee_sats" in q.json()

    job = client.post(
        "/pay/v1/jobs",
        json={"type": "finetune", "preset": "smoke", "dry_run": True},
        headers={"Authorization": f"Bearer {body['token']}"},
    )
    assert job.status_code == 200
    assert job.json()["job"]["status"] == "completed"
