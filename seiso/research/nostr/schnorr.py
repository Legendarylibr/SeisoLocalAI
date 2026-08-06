"""Minimal BIP-340 Schnorr over secp256k1 (Nostr event signatures).

Pure-Python so optional Nostr support does not require a native coincurve build.
"""

from __future__ import annotations

import hashlib
import os

# secp256k1 curve parameters
_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
_GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
_GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
_G = (_GX, _GY)


def _mod_inv(a: int, m: int) -> int:
    return pow(a % m, -1, m)


def _point_add(p1: tuple[int, int] | None, p2: tuple[int, int] | None) -> tuple[int, int] | None:
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2 and (y1 + y2) % _P == 0:
        return None
    if p1 == p2:
        lam = (3 * x1 * x1 * _mod_inv(2 * y1, _P)) % _P
    else:
        lam = ((y2 - y1) * _mod_inv((x2 - x1) % _P, _P)) % _P
    x3 = (lam * lam - x1 - x2) % _P
    y3 = (lam * (x1 - x3) - y1) % _P
    return x3, y3


def _point_mul(k: int, point: tuple[int, int] = _G) -> tuple[int, int] | None:
    k = k % _N
    if k == 0:
        return None
    result: tuple[int, int] | None = None
    addend: tuple[int, int] | None = point
    while k:
        if k & 1:
            result = _point_add(result, addend)
        addend = _point_add(addend, addend)
        k >>= 1
    return result


def _tagged_hash(tag: str, *payloads: bytes) -> bytes:
    tag_hash = hashlib.sha256(tag.encode("utf-8")).digest()
    h = hashlib.sha256()
    h.update(tag_hash)
    h.update(tag_hash)
    for payload in payloads:
        h.update(payload)
    return h.digest()


def _bytes32(i: int) -> bytes:
    return i.to_bytes(32, "big")


def _int_from_bytes(b: bytes) -> int:
    return int.from_bytes(b, "big")


def _has_even_y(point: tuple[int, int]) -> bool:
    return point[1] % 2 == 0


def _xonly(point: tuple[int, int]) -> bytes:
    return _bytes32(point[0])


def _lift_x(x: int) -> tuple[int, int] | None:
    if x >= _P:
        return None
    y_sq = (pow(x, 3, _P) + 7) % _P
    y = pow(y_sq, (_P + 1) // 4, _P)
    if pow(y, 2, _P) != y_sq:
        return None
    if y % 2 != 0:
        y = _P - y
    return x, y


def pubkey_xonly_from_secret(secret: bytes) -> bytes:
    if len(secret) != 32:
        raise ValueError("secret must be 32 bytes")
    d0 = _int_from_bytes(secret)
    if not (1 <= d0 <= _N - 1):
        raise ValueError("invalid secret key")
    point = _point_mul(d0)
    if point is None:
        raise ValueError("invalid secret key")
    return _xonly(point)


def sign_schnorr(secret: bytes, message: bytes, aux_rand: bytes | None = None) -> bytes:
    """BIP-340 sign; ``message`` must be 32 bytes (Nostr event id)."""
    if len(secret) != 32 or len(message) != 32:
        raise ValueError("secret and message must be 32 bytes")
    d0 = _int_from_bytes(secret)
    if not (1 <= d0 <= _N - 1):
        raise ValueError("invalid secret key")
    p = _point_mul(d0)
    if p is None:
        raise ValueError("invalid secret key")
    d = d0 if _has_even_y(p) else (_N - d0)
    pubkey = _xonly(p)
    if aux_rand is None:
        aux_rand = os.urandom(32)
    if len(aux_rand) != 32:
        raise ValueError("aux_rand must be 32 bytes")
    t = _bytes32(d ^ _int_from_bytes(_tagged_hash("BIP0340/aux", aux_rand)))
    k0 = _int_from_bytes(_tagged_hash("BIP0340/nonce", t, pubkey, message)) % _N
    if k0 == 0:
        raise RuntimeError("schnorr nonce failure")
    r_point = _point_mul(k0)
    if r_point is None:
        raise RuntimeError("schnorr nonce failure")
    k = k0 if _has_even_y(r_point) else (_N - k0)
    r = _xonly(r_point)
    e = _int_from_bytes(_tagged_hash("BIP0340/challenge", r, pubkey, message)) % _N
    sig = r + _bytes32((k + e * d) % _N)
    if not verify_schnorr(pubkey, message, sig):
        raise RuntimeError("schnorr self-check failed")
    return sig


def verify_schnorr(pubkey_xonly: bytes, message: bytes, signature: bytes) -> bool:
    if len(pubkey_xonly) != 32 or len(message) != 32 or len(signature) != 64:
        return False
    p = _lift_x(_int_from_bytes(pubkey_xonly))
    if p is None:
        return False
    r = _int_from_bytes(signature[:32])
    s = _int_from_bytes(signature[32:])
    if r >= _P or s >= _N:
        return False
    e = (
        _int_from_bytes(_tagged_hash("BIP0340/challenge", signature[:32], pubkey_xonly, message))
        % _N
    )
    # R = s*G - e*P
    sg = _point_mul(s)
    ep = _point_mul(e, p)
    if ep is None or sg is None:
        return False
    # Negate ep
    ep_neg = (ep[0], _P - ep[1]) if ep[1] != 0 else ep
    r_point = _point_add(sg, ep_neg)
    return r_point is not None and _has_even_y(r_point) and r_point[0] == r
