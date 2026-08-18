"""x402 EVM HTTP 402 funding rail — sim, fail-closed live, exact-scheme checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from seiso.pay.x402 import (
    DEFAULT_NETWORK,
    LIVE_NOT_READY_MSG,
    USDC_BY_NETWORK,
    complete_fund,
    decode_payment_header,
    encode_payment_header,
    is_evm_address,
    mint_fund_challenge,
    normalize_evm_address,
    require_x402_ready,
    sats_to_usdc_atomic,
    x402_advertised,
    x402_asset,
    x402_network,
    x402_sim_enabled,
)


@pytest.fixture()
def pay_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SEISO_ALLOW_PAY", "1")
    monkeypatch.setenv("SEISO_PAY_FAUCET", "1")
    monkeypatch.setenv("SEISO_PAY_X402_SIM", "1")
    monkeypatch.setenv("SEISO_PROTOCOL_FEE_BPS", "500")
    monkeypatch.setenv("SEISO_PROTOCOL_TREASURY_ARK", "ark1testtreasury")
    monkeypatch.setenv("SEISO_OPERATOR_ARK", "ark1testoperator")
    return tmp_path / "data"


def test_evm_address_validation() -> None:
    assert is_evm_address("0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913")
    assert is_evm_address("0x0000000000000000000000000000000000000402")
    assert not is_evm_address("ark1abc")
    assert not is_evm_address("0x123")
    assert not is_evm_address("833589fCD6eDb6E08f4c7C32D4f71b54bdA02913")
    assert normalize_evm_address("0xABCDef0000000000000000000000000000000001") == (
        "0xabcdef0000000000000000000000000000000001"
    )
    with pytest.raises(ValueError, match="invalid EVM address"):
        normalize_evm_address("not-an-address")


def test_default_network_is_base_sepolia() -> None:
    assert x402_network() == DEFAULT_NETWORK
    assert x402_asset() == USDC_BY_NETWORK[DEFAULT_NETWORK]


def test_custom_network_and_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEISO_PAY_X402_NETWORK", "eip155:8453")
    assert x402_network() == "eip155:8453"
    assert x402_asset() == USDC_BY_NETWORK["eip155:8453"]
    monkeypatch.setenv("SEISO_PAY_X402_ASSET", "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")
    assert x402_asset() == "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"


def test_unknown_network_requires_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEISO_PAY_X402_NETWORK", "eip155:999")
    monkeypatch.delenv("SEISO_PAY_X402_ASSET", raising=False)
    with pytest.raises(ValueError, match="no default USDC"):
        x402_asset()


def test_sats_to_usdc_atomic_identity_and_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SEISO_PAY_X402_ATOMIC_PER_SAT", raising=False)
    assert sats_to_usdc_atomic(20_000) == 20_000
    monkeypatch.setenv("SEISO_PAY_X402_ATOMIC_PER_SAT", "2")
    assert sats_to_usdc_atomic(100) == 200
    with pytest.raises(ValueError):
        sats_to_usdc_atomic(0)


def test_header_roundtrip() -> None:
    payload = {"x402Version": 2, "scheme": "exact", "accepts": [{"network": "eip155:84532"}]}
    encoded = encode_payment_header(payload)
    assert "{" not in encoded
    assert decode_payment_header(encoded) == payload
    assert decode_payment_header(encode_payment_header(payload))["x402Version"] == 2


def test_x402_advertised_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEISO_PAY_X402", raising=False)
    assert x402_advertised() is True
    monkeypatch.setenv("SEISO_PAY_X402", "0")
    assert x402_advertised() is False


def test_x402_sim_follows_faucet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEISO_PAY_X402_SIM", raising=False)
    monkeypatch.setenv("SEISO_PAY_FAUCET", "1")
    assert x402_sim_enabled() is True
    monkeypatch.setenv("SEISO_PAY_X402_SIM", "0")
    assert x402_sim_enabled() is False


def test_live_x402_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEISO_PAY_FAUCET", raising=False)
    monkeypatch.setenv("SEISO_PAY_X402", "1")
    monkeypatch.setenv("SEISO_PAY_X402_SIM", "0")
    with pytest.raises(RuntimeError, match="not functional yet"):
        require_x402_ready()
    assert "x402.org" in LIVE_NOT_READY_MSG


def test_mint_and_complete_credits_session(pay_env: Path) -> None:
    from seiso.pay.store import create_session, load_session

    created = create_session(scopes=["inference"], data_dir=pay_env)
    challenge = mint_fund_challenge(
        session_id=created["session_id"],
        amount_sats=12_000,
        data_dir=pay_env,
    )
    assert challenge["method"] == "x402"
    assert challenge["http_status"] == 402
    assert challenge["scheme"] == "exact"
    assert challenge["network"] == DEFAULT_NETWORK
    assert challenge["amount_sats"] == 12_000
    assert challenge["amount_usdc_atomic"] == 12_000
    assert challenge["pay_to"].startswith("0x")
    assert challenge["payment_required"]["x402Version"] == 2
    accepts = challenge["payment_required"]["accepts"]
    assert accepts[0]["scheme"] == "exact"
    assert accepts[0]["maxAmountRequired"] == "12000"
    assert accepts[0]["asset"] == USDC_BY_NETWORK[DEFAULT_NETWORK]
    assert challenge["www_authenticate"].startswith("X402 ")
    assert challenge["sim_payment_signature"]

    decoded = decode_payment_header(challenge["payment_required_header"])
    assert decoded["accepts"][0]["payTo"] == challenge["pay_to"]

    done = complete_fund(
        payment_signature=challenge["sim_payment_signature"],
        data_dir=pay_env,
    )
    assert done["funding_mode"] == "x402"
    assert done["amount_sats"] == 12_000
    session = load_session(created["session_id"], data_dir=pay_env)
    assert session["balance_sats"] == 12_000
    assert session["funding_mode"] == "x402"


def test_complete_is_idempotent_second_call_conflicts(pay_env: Path) -> None:
    from seiso.pay.store import create_session

    created = create_session(scopes=["finetune"], data_dir=pay_env)
    challenge = mint_fund_challenge(
        session_id=created["session_id"],
        amount_sats=500,
        data_dir=pay_env,
    )
    complete_fund(
        payment_signature=challenge["sim_payment_signature"],
        data_dir=pay_env,
    )
    with pytest.raises(RuntimeError, match="already settled"):
        complete_fund(
            payment_signature=challenge["sim_payment_signature"],
            data_dir=pay_env,
        )


def test_wrong_signature_rejected(pay_env: Path) -> None:
    from seiso.pay.store import create_session

    created = create_session(scopes=["inference"], data_dir=pay_env)
    challenge = mint_fund_challenge(
        session_id=created["session_id"],
        amount_sats=100,
        data_dir=pay_env,
    )
    payload = challenge["sim_payment_payload"]
    payload["payload"]["signature"] = "0xdeadbeef" + "00" * 28
    with pytest.raises(ValueError, match="invalid x402 payment signature"):
        complete_fund(payload=payload, data_dir=pay_env)


def test_value_must_be_exact(pay_env: Path) -> None:
    from seiso.pay.store import create_session

    created = create_session(scopes=["inference"], data_dir=pay_env)
    challenge = mint_fund_challenge(
        session_id=created["session_id"],
        amount_sats=100,
        data_dir=pay_env,
    )
    payload = challenge["sim_payment_payload"]
    payload["payload"]["authorization"]["value"] = "99"
    with pytest.raises(ValueError, match="exactly match"):
        complete_fund(payload=payload, data_dir=pay_env)


def test_pay_to_must_match(pay_env: Path) -> None:
    from seiso.pay.store import create_session

    created = create_session(scopes=["inference"], data_dir=pay_env)
    challenge = mint_fund_challenge(
        session_id=created["session_id"],
        amount_sats=100,
        data_dir=pay_env,
    )
    payload = challenge["sim_payment_payload"]
    payload["payload"]["authorization"]["to"] = "0x1111111111111111111111111111111111111111"
    with pytest.raises(ValueError, match="payTo"):
        complete_fund(payload=payload, data_dir=pay_env)


def test_network_mismatch_rejected(pay_env: Path) -> None:
    from seiso.pay.store import create_session

    created = create_session(scopes=["inference"], data_dir=pay_env)
    challenge = mint_fund_challenge(
        session_id=created["session_id"],
        amount_sats=100,
        data_dir=pay_env,
    )
    payload = challenge["sim_payment_payload"]
    payload["network"] = "eip155:1"
    with pytest.raises(ValueError, match="network mismatch"):
        complete_fund(payload=payload, data_dir=pay_env)


def test_operator_evm_used_as_pay_to(pay_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEISO_OPERATOR_EVM", "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913")
    from seiso.pay.store import create_session

    created = create_session(scopes=["inference"], data_dir=pay_env)
    challenge = mint_fund_challenge(
        session_id=created["session_id"],
        amount_sats=10,
        data_dir=pay_env,
    )
    assert challenge["pay_to"] == "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"


def test_http_x402_fund_exchange(pay_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    monkeypatch.setenv("SEISO_PAY_X402_SIM", "1")
    from fastapi.testclient import TestClient

    from seiso.pay.app import build_app

    client = TestClient(build_app())
    wk = client.get("/.well-known/seiso-pay.json")
    assert wk.status_code == 200
    body_wk = wk.json()
    method_ids = {m["id"] for m in body_wk["payment_methods"]}
    assert "x402" in method_ids
    assert body_wk["x402_sim"] is True
    assert "fund_x402" in body_wk["endpoints"]
    assert "x402" in body_wk["payment_methods_note"].lower()

    created = client.post("/pay/v1/sessions", json={"scopes": ["finetune"], "sats": 0})
    token = created.json()["token"]
    session_id = created.json()["session"]["session_id"]
    funding = created.json()["funding"]
    assert funding["x402"]["method"] == "x402"
    assert funding["x402"]["do_not_use_live_evm"] is True

    challenge = client.post(
        "/pay/v1/sessions/fund/x402",
        json={"session_id": session_id, "sats": 8_000},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert challenge.status_code == 402
    assert challenge.headers["WWW-Authenticate"].startswith("X402 ")
    assert "PAYMENT-REQUIRED" in challenge.headers
    ch = challenge.json()
    assert ch["sim_payment_signature"]

    unauth = client.post(
        "/pay/v1/sessions/fund/x402",
        json={"session_id": session_id, "sats": 1},
    )
    assert unauth.status_code == 401

    wrong = client.post(
        "/pay/v1/sessions/fund/x402",
        json={"session_id": "not-mine", "sats": 1},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert wrong.status_code == 403

    done = client.post(
        "/pay/v1/sessions/fund/x402/complete",
        headers={"PAYMENT-SIGNATURE": ch["sim_payment_signature"]},
    )
    assert done.status_code == 200
    assert done.json()["funding_mode"] == "x402"
    assert done.json()["session"]["balance_sats"] == 8_000

    again = client.post(
        "/pay/v1/sessions/fund/x402/complete",
        headers={"PAYMENT-SIGNATURE": ch["sim_payment_signature"]},
    )
    assert again.status_code == 409


def test_hide_x402_from_discovery(pay_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from seiso.pay.ark import funding_instructions
    from seiso.pay.flags import payment_methods

    monkeypatch.setenv("SEISO_PAY_X402", "0")
    ids = {m["id"] for m in payment_methods()}
    assert "x402" not in ids
    assert "ark" in ids
    funding = funding_instructions("sess-test", 1000)
    assert funding["x402"] is None


def test_cli_x402_fund(pay_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEISO_ALLOW_PAY", "1")
    monkeypatch.setenv("SEISO_PAY_X402_SIM", "1")
    monkeypatch.setenv("SEISO_DATA_DIR", str(pay_env))
    from seiso.pay.store import create_session, load_session
    from seiso_cli.commands.pay import pay_session

    created = create_session(scopes=["inference"], data_dir=pay_env)
    pay_session(
        action="fund",
        scopes="inference",
        sats=3_000,
        session=created["session_id"],
        token=None,
        faucet=False,
        l402=False,
        x402=True,
        json_out=True,
    )
    record = load_session(created["session_id"], data_dir=pay_env)
    assert record["balance_sats"] == 3_000
    assert record["funding_mode"] == "x402"
