"""Bitcoin HTLC + Ark voucher builders."""

from __future__ import annotations

import hashlib

import pytest

from seiso.pay.btc.ark_voucher import build_voucher, miniscript
from seiso.pay.btc.htlc import (
    OP_CHECKLOCKTIMEVERIFY,
    OP_SHA256,
    build_htlc,
    buyer_refund_witness,
    encode_locktime,
    operator_witness,
    payment_hash_from_preimage,
    redeem_script,
)


def _pub(n: int) -> bytes:
    return bytes([0x02]) + bytes([n]) * 32


def test_payment_hash() -> None:
    pre = hashlib.sha256(b"secret-preimage-pad-pad-pad-pad!!").digest()
    assert len(pre) == 32
    assert payment_hash_from_preimage(pre) == hashlib.sha256(pre).digest()
    with pytest.raises(ValueError):
        payment_hash_from_preimage(b"short")


def test_htlc_script_contains_hash_and_cltv() -> None:
    pre = bytes(range(32))
    ph = payment_hash_from_preimage(pre)
    script = redeem_script(
        payment_hash=ph,
        operator_pubkey=_pub(1),
        buyer_pubkey=_pub(2),
        locktime=900_000,
    )
    assert bytes([OP_SHA256]) in script
    assert ph in script
    assert _pub(1) in script
    assert _pub(2) in script
    assert bytes([OP_CHECKLOCKTIMEVERIFY]) in script
    offer = build_htlc(
        payment_hash=ph,
        operator_pubkey=_pub(1),
        buyer_pubkey=_pub(2),
        locktime=900_000,
        amount_sats=21_000,
    )
    assert offer.script_pubkey_hex.startswith("0020")
    assert len(bytes.fromhex(offer.witness_program_hex)) == 32
    assert offer.amount_sats == 21_000


def test_witness_stacks() -> None:
    redeem = b"\x63"
    pre = bytes(32)
    op_w = operator_witness(b"\x30\x44", pre, redeem)
    assert op_w[-1] == redeem
    assert op_w[1] == pre
    rf = buyer_refund_witness(b"\x30\x44", redeem)
    assert rf[1] == b""


def test_locktime_scriptnum() -> None:
    assert encode_locktime(0) == b"\x00"
    assert encode_locktime(1) == b"\x01"
    assert encode_locktime(255) == b"\xff\x00"  # high bit set → extra 0


def test_ark_miniscript() -> None:
    ph = "ab" * 32
    desc = miniscript(
        operator_xonly_or_desc="operatorkey",
        buyer_xonly_or_desc="buyerkey",
        payment_hash_hex=ph,
        locktime=500_000,
    )
    assert "sha256(" + ph + ")" in desc
    assert "after(500000)" in desc
    v = build_voucher(
        amount_sats=1000,
        payment_hash_hex=ph,
        operator_key="operatorkey",
        buyer_key="buyerkey",
        locktime=500_000,
    )
    assert "not a live ark" in v.note.lower()
    with pytest.raises(ValueError):
        build_voucher(
            amount_sats=0,
            payment_hash_hex=ph,
            operator_key="a",
            buyer_key="b",
            locktime=1,
        )
