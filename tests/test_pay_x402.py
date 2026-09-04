"""Tests for x402 EVM funding rail and per-request payments."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _x402_env(monkeypatch) -> None:
    """Enable x402 sim mode for all tests."""
    monkeypatch.setenv("SEISO_PAY_X402_SIM", "1")
    monkeypatch.setenv("SEISO_PAY_X402_NETWORK", "eip155:84532")
    monkeypatch.setenv("SEISO_DATA_DIR", "/tmp/.seiso-test-x402")
    monkeypatch.setenv("SEISO_PAY_X402_ROOT_KEY", "test-root-key-for-deterministic-hmac")


def _clean_data() -> None:
    import shutil

    d = Path("/tmp/.seiso-test-x402")
    if d.exists():
        shutil.rmtree(str(d))


def test_x402_advertised_default() -> None:
    from seiso.pay.x402 import x402_advertised

    assert x402_advertised() is True


def test_x402_sim_enabled() -> None:
    from seiso.pay.x402 import x402_sim_enabled

    assert x402_sim_enabled() is True


def test_x402_sim_disabled(monkeypatch) -> None:
    monkeypatch.setenv("SEISO_PAY_X402_SIM", "0")
    from seiso.pay.x402 import x402_sim_enabled

    assert x402_sim_enabled() is False


def test_x402_ready() -> None:
    from seiso.pay.x402 import x402_ready

    ok, mode = x402_ready()
    assert ok is True
    assert mode == "sim"


def test_is_evm_address() -> None:
    from seiso.pay.x402 import is_evm_address

    assert is_evm_address("0x1234567890123456789012345678901234567890") is True
    assert is_evm_address("0x123456789012345678901234567890123456789") is False  # 41 chars
    assert is_evm_address(None) is False
    assert is_evm_address("") is False
    assert is_evm_address("not-an-address") is False


def test_normalize_evm_address() -> None:
    from seiso.pay.x402 import normalize_evm_address

    result = normalize_evm_address("0x1234567890123456789012345678901234567890")
    assert result == "0x1234567890123456789012345678901234567890"
    # Mixed case
    result2 = normalize_evm_address("0xAbCdEf1234567890123456789012345678901234")
    assert result2 == "0xabcdef1234567890123456789012345678901234"


def test_list_supported_networks() -> None:
    from seiso.pay.x402 import list_supported_networks

    nets = list_supported_networks()
    assert len(nets) > 30
    # Check for key chains
    caip2s = {n["caip2"] for n in nets}
    assert "eip155:1" in caip2s  # Ethereum
    assert "eip155:8453" in caip2s  # Base
    assert "eip155:42161" in caip2s  # Arbitrum
    assert "eip155:4663" in caip2s  # Robinhood Chain
    assert "eip155:46630" in caip2s  # Robinhood Chain Testnet


def test_x402_network_default() -> None:
    from seiso.pay.x402 import x402_network

    assert x402_network() == "eip155:84532"


def test_x402_asset_base_sepolia() -> None:
    from seiso.pay.x402 import x402_asset

    asset = x402_asset("eip155:84532")
    assert asset.startswith("0x")
    assert len(asset) == 42


def test_x402_asset_robinhood() -> None:
    from seiso.pay.x402 import USDC_BY_NETWORK

    assert "eip155:4663" in USDC_BY_NETWORK


def test_sats_to_usdc_atomic() -> None:
    from seiso.pay.x402 import sats_to_usdc_atomic

    result = sats_to_usdc_atomic(1000)
    assert result == 1000  # default 1:1 mapping


def test_mint_and_complete_fund(monkeypatch) -> None:
    _clean_data()
    from seiso.pay.store import create_session

    session = create_session(scopes=["inference"], data_dir=Path("/tmp/.seiso-test-x402"))
    sid = session["session_id"]

    from seiso.pay.x402 import complete_fund, mint_fund_challenge

    chal = mint_fund_challenge(
        session_id=sid,
        amount_sats=5000,
        data_dir=Path("/tmp/.seiso-test-x402"),
    )
    assert chal["method"] == "x402"
    assert chal["status"] == "ready"
    assert chal["amount_sats"] == 5000
    assert chal["http_status"] == 402
    assert "sim_payment_signature" in chal

    # Complete the fund
    sig = chal["sim_payment_signature"]
    result = complete_fund(
        payment_signature=sig,
        data_dir=Path("/tmp/.seiso-test-x402"),
    )
    assert result["funding_mode"] == "x402"
    assert result["amount_sats"] == 5000
    assert result["session"]["balance_sats"] >= 5000

    # Verify double-settle rejected
    with pytest.raises(RuntimeError, match="already settled"):
        complete_fund(payment_signature=sig, data_dir=Path("/tmp/.seiso-test-x402"))


def test_mint_with_custom_network(monkeypatch) -> None:
    _clean_data()
    from seiso.pay.store import create_session
    from seiso.pay.x402 import mint_fund_challenge

    session = create_session(scopes=["inference"], data_dir=Path("/tmp/.seiso-test-x402"))

    # Fund on Robinhood Chain
    chal = mint_fund_challenge(
        session_id=session["session_id"],
        amount_sats=1000,
        data_dir=Path("/tmp/.seiso-test-x402"),
        network="eip155:4663",
    )
    assert chal["network"] == "eip155:4663"
    assert chal["amount_sats"] == 1000


def test_funding_x402_block() -> None:
    from seiso.pay.x402 import funding_x402_block

    block = funding_x402_block("test-session-123", 5000)
    assert block is not None
    assert block["method"] == "x402"
    assert block["status"] == "ready"
    assert block["session_id"] == "test-session-123"


def test_funding_x402_block_not_advertised(monkeypatch) -> None:
    monkeypatch.setenv("SEISO_PAY_X402", "0")
    from seiso.pay.x402 import funding_x402_block

    block = funding_x402_block("test-session-123", 5000)
    assert block is None


def test_encode_decode_payment_header() -> None:
    from seiso.pay.x402 import decode_payment_header, encode_payment_header

    payload = {"test": "value", "x402Version": 2}
    encoded = encode_payment_header(payload)
    decoded = decode_payment_header(encoded)
    assert decoded["test"] == "value"
    assert decoded["x402Version"] == 2


def test_www_authenticate_header() -> None:
    from seiso.pay.x402 import www_authenticate_header

    h = www_authenticate_header(
        network="eip155:84532", pay_to="0x1234567890123456789012345678901234567890"
    )
    assert "X402" in h
    assert "eip155:84532" in h
    assert "0x1234567890" in h


def test_payment_methods_includes_x402(monkeypatch) -> None:
    from seiso.pay.flags import payment_methods

    methods = payment_methods()
    x402 = [m for m in methods if m["id"] == "x402"]
    assert len(x402) == 1
    assert x402[0]["status"] == "sim"


def test_payment_methods_x402_hidden(monkeypatch) -> None:
    monkeypatch.setenv("SEISO_PAY_X402", "0")
    from seiso.pay.flags import payment_methods

    methods = payment_methods()
    x402 = [m for m in methods if m["id"] == "x402"]
    assert len(x402) == 0


def test_ark_funding_instructions_includes_x402(monkeypatch) -> None:
    monkeypatch.setenv("SEISO_PAY_X402_SIM", "1")
    from seiso.pay.ark import funding_instructions

    instr = funding_instructions("test-session-ark", 1000)
    assert "x402" in instr
    assert instr["x402"] is not None


def test_list_supported_networks_structure() -> None:
    from seiso.pay.x402 import list_supported_networks

    nets = list_supported_networks()
    for n in nets:
        assert "caip2" in n
        assert "name" in n
        assert "usdc" in n
        assert n["usdc"].startswith("0x")


def test_operator_evm_default() -> None:
    from seiso.pay.x402 import operator_evm

    addr = operator_evm()
    assert addr.startswith("0x")
    assert len(addr) == 42


def test_operator_evm_env(monkeypatch) -> None:
    monkeypatch.setenv("SEISO_OPERATOR_EVM", "0x1234567890123456789012345678901234567890")
    from seiso.pay.x402 import operator_evm

    assert operator_evm() == "0x1234567890123456789012345678901234567890"


def test_protocol_treasury_evm(monkeypatch) -> None:
    monkeypatch.setenv("SEISO_PROTOCOL_TREASURY_EVM", "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    from seiso.pay.x402 import protocol_treasury_evm

    assert protocol_treasury_evm() == "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def test_protocol_treasury_evm_empty() -> None:
    from seiso.pay.x402 import protocol_treasury_evm

    assert protocol_treasury_evm() == ""
