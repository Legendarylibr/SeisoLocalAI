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


def test_pay_quote_cli_requires_opt_in(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``seiso pay quote`` must refuse without SEISO_ALLOW_PAY (experimental)."""
    monkeypatch.delenv("SEISO_ALLOW_PAY", raising=False)
    import typer

    from seiso_cli.commands.pay import pay_quote

    with pytest.raises(typer.Exit) as exc_info:
        pay_quote(
            type_="finetune",
            preset="smoke",
            prompt_tokens=0,
            completion_tokens=0,
            flat_call=False,
        )
    assert exc_info.value.exit_code == 1
    captured = capsys.readouterr()
    err = captured.out + captured.err
    assert "SEISO_ALLOW_PAY" in err
    assert "experimental" in err.lower()


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
    body_wk = wk.json()
    assert body_wk["protocol_fee_bps"] == 500
    method_ids = {m["id"] for m in body_wk["payment_methods"]}
    assert method_ids >= {"ark", "l402", "faucet"}
    assert "L402" in body_wk["payment_methods_note"]
    assert body_wk["l402_sim"] is True
    assert "fund_l402" in body_wk["endpoints"]

    created = client.post(
        "/pay/v1/sessions",
        json={"scopes": ["finetune"], "sats": 20_000},
    )
    assert created.status_code == 200
    body = created.json()
    assert body["token"].startswith("seiso_pay_")
    assert body["session"]["status"] == "active"
    funding = body["funding"]
    assert "payment_methods" in funding
    assert funding["l402"]["method"] == "l402"
    assert funding["l402"]["status"] == "ready"
    assert funding["l402"]["do_not_use_live_ln"] is True
    assert "lightningfaucet.com" in funding["l402"]["reference"]

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


def test_l402_fund_exchange_and_job_failure_refund(
    pay_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L402 sim credits session; failed jobs restore escrow to balance."""
    pytest.importorskip("fastapi")
    monkeypatch.setenv("SEISO_PAY_L402_SIM", "1")
    from fastapi.testclient import TestClient

    from seiso.pay.app import build_app
    from seiso.pay.jobs import _fail_job, job_receipt
    from seiso.pay.pricing import quote_job
    from seiso.pay.store import create_job, escrow_hold, load_session

    client = TestClient(build_app())
    created = client.post(
        "/pay/v1/sessions",
        json={"scopes": ["finetune"], "sats": 0},
    )
    assert created.status_code == 200
    token = created.json()["token"]
    session_id = created.json()["session"]["session_id"]
    assert created.json()["session"]["status"] == "pending"

    challenge = client.post(
        "/pay/v1/sessions/fund/l402",
        json={"session_id": session_id, "sats": 25_000},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert challenge.status_code == 402
    assert "WWW-Authenticate" in challenge.headers
    assert challenge.headers["WWW-Authenticate"].startswith("L402 ")
    ch = challenge.json()
    assert ch["macaroon"]
    assert ch["invoice"].startswith("lnbcsseisosim1")
    assert ch["sim_preimage"]

    unauth = client.post(
        "/pay/v1/sessions/fund/l402",
        json={"session_id": session_id, "sats": 1_000},
    )
    assert unauth.status_code == 401

    wrong = client.post(
        "/pay/v1/sessions/fund/l402",
        json={"session_id": "not-my-session", "sats": 1_000},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert wrong.status_code == 403

    done = client.post(
        "/pay/v1/sessions/fund/l402/complete",
        headers={"Authorization": f"L402 {ch['macaroon']}:{ch['sim_preimage']}"},
    )
    assert done.status_code == 200
    assert done.json()["session"]["status"] == "active"
    assert done.json()["session"]["balance_sats"] == 25_000
    assert done.json()["funding_mode"] == "l402"

    again = client.post(
        "/pay/v1/sessions/fund/l402/complete",
        headers={"Authorization": f"L402 {ch['macaroon']}:{ch['sim_preimage']}"},
    )
    assert again.status_code == 409

    me = client.get(
        "/pay/v1/sessions/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me.json()["funding_mode"] == "l402"

    quote = quote_job("finetune", preset="smoke")
    held = create_job(
        session_id=session_id,
        job_type="finetune",
        quote=quote,
        preset="smoke",
    )
    escrow_hold(
        session_id,
        total_sats=int(quote["total_sats"]),
        job_id=held["job_id"],
    )
    held["status"] = "running"
    before = load_session(session_id)["balance_sats"]
    failed = _fail_job(held, "simulated trainer crash", data_dir=None)
    assert failed["status"] == "failed"
    assert failed["refunded_sats"] == quote["total_sats"]
    assert failed["settlement"]["status"] == "refunded"
    assert job_receipt(failed)["refunded_sats"] == quote["total_sats"]
    after = load_session(session_id)["balance_sats"]
    assert after == before + int(quote["total_sats"])
    assert load_session(session_id)["refunded_sats"] == quote["total_sats"]


def test_l402_hide_and_fail_closed(pay_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from seiso.pay.ark import funding_instructions
    from seiso.pay.flags import payment_methods
    from seiso.pay.l402 import require_l402_ready

    monkeypatch.setenv("SEISO_PAY_L402", "0")
    ids = {m["id"] for m in payment_methods()}
    assert "l402" not in ids
    assert "ark" in ids
    funding = funding_instructions("sess-test", 1000)
    assert funding["l402"] is None

    monkeypatch.delenv("SEISO_PAY_FAUCET", raising=False)
    monkeypatch.setenv("SEISO_PAY_L402_SIM", "0")
    monkeypatch.setenv("SEISO_PAY_L402", "1")
    with pytest.raises(RuntimeError, match="not functional yet"):
        require_l402_ready()


def test_cancel_job_refunds(pay_env: Path) -> None:
    from seiso.pay.jobs import cancel_job
    from seiso.pay.pricing import quote_job
    from seiso.pay.store import (
        activate_session,
        create_job,
        create_session,
        escrow_hold,
        load_session,
        save_job,
    )

    created = create_session(scopes=["finetune"])
    activate_session(created["session_id"], amount_sats=50_000, funding_mode="faucet")
    quote = quote_job("finetune", preset="smoke")
    job = create_job(
        session_id=created["session_id"],
        job_type="finetune",
        quote=quote,
        preset="smoke",
    )
    escrow_hold(
        created["session_id"],
        total_sats=int(quote["total_sats"]),
        job_id=job["job_id"],
    )
    job["status"] = "running"
    save_job(job)
    before = load_session(created["session_id"])["balance_sats"]
    cancelled = cancel_job(job["job_id"])
    assert cancelled["status"] == "cancelled"
    assert cancelled["refunded_sats"] == quote["total_sats"]
    assert cancelled["settlement"]["status"] == "refunded"
    again = cancel_job(job["job_id"])
    assert again["refunded_sats"] == quote["total_sats"]
    after = load_session(created["session_id"])["balance_sats"]
    assert after == before + int(quote["total_sats"])


def test_l402_fund_id_idempotent_no_double_credit(pay_env: Path) -> None:
    from seiso.pay.l402 import complete_fund, mint_fund_challenge
    from seiso.pay.store import create_session, load_session

    created = create_session(scopes=["inference"])
    challenge = mint_fund_challenge(session_id=created["session_id"], amount_sats=5_000)
    complete_fund(
        macaroon=str(challenge["macaroon"]),
        preimage_hex=str(challenge["sim_preimage"]),
    )
    session = load_session(created["session_id"])
    assert session["balance_sats"] == 5_000
    assert challenge["challenge_id"] in session["fund_ids"]

    # Simulate crash/retry: credit again with same fund_id via activate_session.
    from seiso.pay.store import activate_session

    activate_session(
        created["session_id"],
        amount_sats=5_000,
        funding_mode="l402",
        fund_id=str(challenge["challenge_id"]),
    )
    assert load_session(created["session_id"])["balance_sats"] == 5_000


def test_escrow_refund_idempotent_by_job_id(pay_env: Path) -> None:
    from seiso.pay.store import (
        activate_session,
        create_session,
        escrow_hold,
        escrow_release_refund,
        load_session,
    )

    created = create_session(scopes=["finetune"])
    activate_session(created["session_id"], amount_sats=10_000, funding_mode="faucet")
    escrow_hold(created["session_id"], total_sats=1_000, job_id="job-a")
    before = load_session(created["session_id"])["balance_sats"]
    escrow_release_refund(created["session_id"], amount_sats=1_000, job_id="job-a", reason="test")
    escrow_release_refund(created["session_id"], amount_sats=1_000, job_id="job-a", reason="test")
    assert load_session(created["session_id"])["balance_sats"] == before + 1_000


def test_complete_after_cancel_does_not_settle(pay_env: Path) -> None:
    from seiso.pay.jobs import _complete_job, cancel_job
    from seiso.pay.pricing import quote_job
    from seiso.pay.store import (
        activate_session,
        create_job,
        create_session,
        escrow_hold,
        load_session,
        save_job,
    )

    created = create_session(scopes=["finetune"])
    activate_session(created["session_id"], amount_sats=50_000, funding_mode="faucet")
    quote = quote_job("finetune", preset="smoke")
    job = create_job(
        session_id=created["session_id"],
        job_type="finetune",
        quote=quote,
        preset="smoke",
    )
    escrow_hold(
        created["session_id"],
        total_sats=int(quote["total_sats"]),
        job_id=job["job_id"],
    )
    job["status"] = "running"
    save_job(job)
    cancelled = cancel_job(job["job_id"])
    assert cancelled["status"] == "cancelled"
    bal_after_cancel = load_session(created["session_id"])["balance_sats"]
    # Runner loses the race: attempt settle after cancel.
    result = _complete_job(job, data_dir=None, dry_run=True)
    assert result["status"] == "cancelled"
    assert result["settlement"]["status"] == "refunded"
    assert load_session(created["session_id"])["balance_sats"] == bal_after_cancel


def test_settle_failure_refunds_escrow(pay_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from seiso.pay.jobs import _complete_job
    from seiso.pay.pricing import quote_job
    from seiso.pay.store import (
        activate_session,
        create_job,
        create_session,
        escrow_hold,
        load_session,
        save_job,
    )

    monkeypatch.setenv("SEISO_ARK_BACKEND", "bark")
    created = create_session(scopes=["finetune"])
    activate_session(created["session_id"], amount_sats=50_000, funding_mode="faucet")
    quote = quote_job("finetune", preset="smoke")
    job = create_job(
        session_id=created["session_id"],
        job_type="finetune",
        quote=quote,
        preset="smoke",
    )
    escrow_hold(
        created["session_id"],
        total_sats=int(quote["total_sats"]),
        job_id=job["job_id"],
    )
    job["status"] = "running"
    save_job(job)
    before_hold = 50_000 - int(quote["total_sats"])
    assert load_session(created["session_id"])["balance_sats"] == before_hold
    failed = _complete_job(job, data_dir=None, dry_run=True)
    assert failed["status"] == "failed"
    assert "settle failed" in (failed.get("error") or "")
    assert failed["settlement"]["status"] == "refunded"
    assert load_session(created["session_id"])["balance_sats"] == 50_000


def test_buyer_config_must_be_under_configs(pay_env: Path) -> None:
    from seiso.pay.jobs import _sandbox_config_path

    with pytest.raises(ValueError, match="configs"):
        _sandbox_config_path("/etc/passwd")
    with pytest.raises(ValueError, match="configs|\\.\\."):
        _sandbox_config_path("../secrets.yaml")


def test_relative_artifact_name_rejects_traversal() -> None:
    from seiso.security import assert_relative_artifact_name

    assert assert_relative_artifact_name("checkpoint-best") == "checkpoint-best"
    with pytest.raises(ValueError, match="\\.\\."):
        assert_relative_artifact_name("../../../models/other")
    with pytest.raises(ValueError, match="relative"):
        assert_relative_artifact_name("/tmp/out")
