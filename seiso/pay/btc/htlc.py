"""Per-request Bitcoin HTLC (P2WSH) for L1 / some L2 escrow.

Lightning L402 remains the preferred L2 rail (invoice per request).
This script is the on-chain fallback: operator claims with preimage;
buyer refunds after ``locktime``.

Script (BIP16 / BIP141 P2WSH):

    OP_IF
      OP_SHA256 <payment_hash> OP_EQUALVERIFY
      <operator_pubkey> OP_CHECKSIG
    OP_ELSE
      <locktime> OP_CHECKLOCKTIMEVERIFY OP_DROP
      <buyer_pubkey> OP_CHECKSIG
    OP_ENDIF

Witness redeem (operator):  <sig> <preimage> OP_1
Witness refund (buyer):     <sig> OP_0

No private keys are generated here — callers supply compressed pubkeys
and a 32-byte payment hash.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import asdict, dataclass
from typing import Any

OP_0 = 0x00
OP_IF = 0x63
OP_ELSE = 0x67
OP_ENDIF = 0x68
OP_DROP = 0x75
OP_EQUALVERIFY = 0x88
OP_SHA256 = 0xA8
OP_CHECKSIG = 0xAC
OP_CHECKLOCKTIMEVERIFY = 0xB1


def _push(data: bytes) -> bytes:
    n = len(data)
    if n < 0x4C:
        return bytes([n]) + data
    if n <= 0xFF:
        return bytes([0x4C, n]) + data
    raise ValueError("push data too large for this HTLC builder")


def _scriptnum(n: int) -> bytes:
    """BIP62 minimally-encoded script number (unsigned locktime)."""
    if n < 0:
        raise ValueError("locktime must be >= 0")
    if n == 0:
        return b"\x00"
    neg = False
    out = bytearray()
    while n:
        out.append(n & 0xFF)
        n >>= 8
    if out[-1] & 0x80:
        out.append(0x80 if neg else 0x00)
    return bytes(out)


def encode_locktime(locktime: int) -> bytes:
    if locktime < 0 or locktime > 0xFFFFFFFF:
        raise ValueError("locktime out of range")
    return _scriptnum(locktime)


def redeem_script(
    *,
    payment_hash: bytes,
    operator_pubkey: bytes,
    buyer_pubkey: bytes,
    locktime: int,
) -> bytes:
    if len(payment_hash) != 32:
        raise ValueError("payment_hash must be 32 bytes (SHA-256)")
    if len(operator_pubkey) not in {33, 65}:
        raise ValueError("operator_pubkey must be compressed (33) or uncompressed (65)")
    if len(buyer_pubkey) not in {33, 65}:
        raise ValueError("buyer_pubkey must be compressed (33) or uncompressed (65)")
    lt = encode_locktime(locktime)
    return b"".join(
        [
            bytes([OP_IF]),
            bytes([OP_SHA256]),
            _push(payment_hash),
            bytes([OP_EQUALVERIFY]),
            _push(operator_pubkey),
            bytes([OP_CHECKSIG]),
            bytes([OP_ELSE]),
            _push(lt),
            bytes([OP_CHECKLOCKTIMEVERIFY]),
            bytes([OP_DROP]),
            _push(buyer_pubkey),
            bytes([OP_CHECKSIG]),
            bytes([OP_ENDIF]),
        ]
    )


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def payment_hash_from_preimage(preimage: bytes) -> bytes:
    if len(preimage) != 32:
        raise ValueError("preimage must be 32 bytes")
    return sha256(preimage)


def witness_script_hash(script: bytes) -> bytes:
    return sha256(script)


def p2wsh_program(script: bytes) -> bytes:
    """SegWit v0 witness program (32-byte SHA256 of redeem script)."""
    return witness_script_hash(script)


def p2wsh_script_pubkey(script: bytes) -> bytes:
    return bytes([0x00, 0x20]) + p2wsh_program(script)


@dataclass(frozen=True, slots=True)
class HtlcOffer:
    payment_hash_hex: str
    operator_pubkey_hex: str
    buyer_pubkey_hex: str
    locktime: int
    redeem_script_hex: str
    witness_program_hex: str
    script_pubkey_hex: str
    amount_sats: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_htlc(
    *,
    payment_hash: bytes,
    operator_pubkey: bytes,
    buyer_pubkey: bytes,
    locktime: int,
    amount_sats: int,
) -> HtlcOffer:
    script = redeem_script(
        payment_hash=payment_hash,
        operator_pubkey=operator_pubkey,
        buyer_pubkey=buyer_pubkey,
        locktime=locktime,
    )
    return HtlcOffer(
        payment_hash_hex=payment_hash.hex(),
        operator_pubkey_hex=operator_pubkey.hex(),
        buyer_pubkey_hex=buyer_pubkey.hex(),
        locktime=int(locktime),
        redeem_script_hex=script.hex(),
        witness_program_hex=p2wsh_program(script).hex(),
        script_pubkey_hex=p2wsh_script_pubkey(script).hex(),
        amount_sats=int(amount_sats),
    )


def operator_witness(sig: bytes, preimage: bytes, redeem: bytes) -> list[bytes]:
    """P2WSH witness stack for the hash-lock (operator) path."""
    if len(preimage) != 32:
        raise ValueError("preimage must be 32 bytes")
    return [sig, preimage, b"\x01", redeem]


def buyer_refund_witness(sig: bytes, redeem: bytes) -> list[bytes]:
    """P2WSH witness stack for the refund path."""
    return [sig, b"", redeem]


def nsequence_for_locktime() -> int:
    """Use 0xfffffffe so CLTV is not disabled (BIP68 / BIP112)."""
    return 0xFFFFFFFE


def locktime_bytes_le(locktime: int) -> bytes:
    return struct.pack("<I", int(locktime) & 0xFFFFFFFF)
