"""Minimal Bech32 (BIP-173) encode/decode for NIP-19 nsec/npub."""

from __future__ import annotations

_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def _polymod(values: list[int]) -> int:
    generators = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for value in values:
        top = chk >> 25
        chk = ((chk & 0x1FFFFFF) << 5) ^ value
        for i in range(5):
            chk ^= generators[i] if ((top >> i) & 1) else 0
    return chk


def _hrp_expand(hrp: str) -> list[int]:
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]


def _create_checksum(hrp: str, data: list[int]) -> list[int]:
    values = _hrp_expand(hrp) + data
    polymod = _polymod(values + [0, 0, 0, 0, 0, 0]) ^ 1
    return [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]


def _verify_checksum(hrp: str, data: list[int]) -> bool:
    return _polymod(_hrp_expand(hrp) + data) == 1


def _convertbits(data: bytes | list[int], from_bits: int, to_bits: int, pad: bool = True) -> list[int]:
    acc = 0
    bits = 0
    ret: list[int] = []
    maxv = (1 << to_bits) - 1
    for value in data:
        if value < 0 or (value >> from_bits):
            raise ValueError("invalid bech32 data")
        acc = (acc << from_bits) | value
        bits += from_bits
        while bits >= to_bits:
            bits -= to_bits
            ret.append((acc >> bits) & maxv)
    if pad:
        if bits:
            ret.append((acc << (to_bits - bits)) & maxv)
    elif bits >= from_bits or ((acc << (to_bits - bits)) & maxv):
        raise ValueError("invalid bech32 padding")
    return ret


def bech32_encode(hrp: str, data: bytes) -> str:
    values = _convertbits(data, 8, 5)
    combined = values + _create_checksum(hrp, values)
    return hrp + "1" + "".join(_CHARSET[d] for d in combined)


def bech32_decode(value: str) -> tuple[str, bytes]:
    raw = (value or "").strip()
    if not raw or any(ord(x) < 33 or ord(x) > 126 for x in raw):
        raise ValueError("invalid bech32 string")
    if raw.lower() != raw and raw.upper() != raw:
        raise ValueError("mixed-case bech32 is invalid")
    raw = raw.lower()
    pos = raw.rfind("1")
    if pos < 1 or pos + 7 > len(raw):
        raise ValueError("invalid bech32 separator")
    hrp = raw[:pos]
    data_part = raw[pos + 1 :]
    try:
        data = [_CHARSET.index(c) for c in data_part]
    except ValueError as exc:
        raise ValueError("invalid bech32 character") from exc
    if not _verify_checksum(hrp, data):
        raise ValueError("invalid bech32 checksum")
    decoded = _convertbits(data[:-6], 5, 8, pad=False)
    return hrp, bytes(decoded)
