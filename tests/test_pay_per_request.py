"""Per-request 402 quotes and sim settlement."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from seiso.pay.oracle import OraclePrices
from seiso.pay.per_request import (
    complete_eth_request,
    complete_sim,
    mint_request_quote,
    request_paid,
    sim_receipt,
)


@pytest.fixture()
def pay_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SEISO_ALLOW_PAY", "1")
    monkeypatch.setenv("SEISO_PAY_FAUCET", "1")
    monkeypatch.setenv("SEISO_PAY_X402_SIM", "1")
    monkeypatch.setenv("SEISO_PAY_L402_SIM", "1")
    monkeypatch.setenv("SEISO_PAY_PER_REQUEST", "1")
    monkeypatch.setenv("SEISO_PAY_ETH_USD_8", str(2500 * 10**8))
    monkeypatch.setenv("SEISO_PAY_BTC_USD_8", str(100_000 * 10**8))
    monkeypatch.setenv("SEISO_PAY_ORACLE_UPDATED_AT", str(time.time()))
    monkeypatch.setenv("SEISO_PROTOCOL_TREASURY_ARK", "ark1t")
    monkeypatch.setenv("SEISO_OPERATOR_ARK", "ark1o")
    return tmp_path / "data"


def test_mint_includes_eth_usdc_l402_ark(pay_env: Path) -> None:
    ch = mint_request_quote(
        prompt_tokens=100,
        completion_tokens=50,
        data_dir=pay_env,
        prices=OraclePrices(
            eth_usd_8=2500 * 10**8,
            btc_usd_8=100_000 * 10**8,
            updated_at=time.time(),
            source="test",
        ),
    )
    assert ch["http_status"] == 402
    assert ch["per_request"] is True
    assert ch["fx"]["wei"] > 0
    assert ch["fx"]["usdc_atomic"] > 0
    assert "eth" in ch["rails"]
    assert ch["rails"]["eth"]["wei"] == ch["fx"]["wei"]
    assert "x402" in ch["rails"]
    assert ch["rails"]["ark"]["method"] == "ark"
    assert request_paid(ch["request_id"], data_dir=pay_env) is False


def test_sim_eth_marks_paid(pay_env: Path) -> None:
    ch = mint_request_quote(
        prompt_tokens=10,
        completion_tokens=10,
        data_dir=pay_env,
        prices=OraclePrices(
            eth_usd_8=2500 * 10**8,
            btc_usd_8=100_000 * 10**8,
            updated_at=time.time(),
            source="test",
        ),
    )
    rid = ch["request_id"]
    rec = complete_eth_request(rid, data_dir=pay_env)
    assert rec["status"] == "paid"
    assert rec["paid_via"] == "eth"
    assert request_paid(rid, data_dir=pay_env) is True
    with pytest.raises(RuntimeError, match="already paid"):
        complete_sim(rid, via="eth", receipt="deadbeef", data_dir=pay_env)


def test_bad_receipt_rejected(pay_env: Path) -> None:
    ch = mint_request_quote(
        prompt_tokens=10,
        completion_tokens=10,
        data_dir=pay_env,
        prices=OraclePrices(
            eth_usd_8=2500 * 10**8,
            btc_usd_8=100_000 * 10**8,
            updated_at=time.time(),
            source="test",
        ),
    )
    with pytest.raises(ValueError, match="invalid"):
        complete_sim(ch["request_id"], via="eth", receipt="00" * 32, data_dir=pay_env)
    good = sim_receipt(ch["request_id"], via="eth", data_dir=pay_env)
    complete_sim(ch["request_id"], via="eth", receipt=good, data_dir=pay_env)


def test_http_per_request_chat_402(pay_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from seiso.pay.app import build_app

    client = TestClient(build_app())
    wk = client.get("/.well-known/seiso-pay.json")
    assert wk.json()["per_request"] is True
    assert "eth" in wk.json()["assets"]

    r = client.post(
        "/pay/v1/requests",
        json={"prompt_tokens": 20, "completion_tokens": 20},
    )
    assert r.status_code == 402
    assert "PAYMENT-REQUIRED" in r.headers
    body = r.json()
    assert body["fx"]["wei"] > 0
    rid = body["request_id"]

    done = client.post(
        f"/pay/v1/requests/{rid}/complete",
        json={"via": "eth"},
    )
    assert done.status_code == 200
    assert done.json()["status"] == "paid"

    chat = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
    )
    assert chat.status_code == 402
    assert chat.json()["per_request"] is True
