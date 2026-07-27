"""NIP-49 private key encryption (ncryptsec).

Uses scrypt + XChaCha20-Poly1305 as specified in
https://github.com/nostr-protocol/nips/blob/master/49.md
"""

from __future__ import annotations

import os
import struct
import unicodedata
from typing import Final

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from seiso.research.nostr.bech32 import bech32_decode, bech32_encode

_PAYLOAD_LEN: Final = 91
_VERSION: Final = 0x02


def _normalize_password(password: str) -> bytes:
    return unicodedata.normalize("NFKC", password).encode("utf-8")


def _scrypt_key(password: str, salt: bytes, log_n: int) -> bytes:
    if not (1 <= log_n <= 22):
        raise ValueError("log_n must be between 1 and 22")
    kdf = Scrypt(
        salt=salt,
        length=32,
        n=2**log_n,
        r=8,
        p=1,
    )
    return kdf.derive(_normalize_password(password))


def _rotl32(v: int, n: int) -> int:
    return ((v << n) | (v >> (32 - n))) & 0xFFFFFFFF


def _quarter_round(state: list[int], a: int, b: int, c: int, d: int) -> None:
    state[a] = (state[a] + state[b]) & 0xFFFFFFFF
    state[d] = _rotl32(state[d] ^ state[a], 16)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] = _rotl32(state[b] ^ state[c], 12)
    state[a] = (state[a] + state[b]) & 0xFFFFFFFF
    state[d] = _rotl32(state[d] ^ state[a], 8)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] = _rotl32(state[b] ^ state[c], 7)


def _hchacha20(key: bytes, nonce16: bytes) -> bytes:
    """HChaCha20 (draft-irtf-cfrg-xchacha) → 32-byte subkey."""
    if len(key) != 32 or len(nonce16) != 16:
        raise ValueError("HChaCha20 requires 32-byte key and 16-byte nonce")
    constants = b"expand 32-byte k"
    state = list(struct.unpack("<16I", constants + key + nonce16))
    for _ in range(10):
        _quarter_round(state, 0, 4, 8, 12)
        _quarter_round(state, 1, 5, 9, 13)
        _quarter_round(state, 2, 6, 10, 14)
        _quarter_round(state, 3, 7, 11, 15)
        _quarter_round(state, 0, 5, 10, 15)
        _quarter_round(state, 1, 6, 11, 12)
        _quarter_round(state, 2, 7, 8, 13)
        _quarter_round(state, 3, 4, 9, 14)
    out = state[0:4] + state[12:16]
    return struct.pack("<8I", *out)


def _xchacha20poly1305_encrypt(
    key: bytes, nonce24: bytes, plaintext: bytes, aad: bytes
) -> bytes:
    if len(nonce24) != 24:
        raise ValueError("XChaCha20-Poly1305 nonce must be 24 bytes")
    subkey = _hchacha20(key, nonce24[:16])
    # libsodium IETF construction: 4 zero bytes + last 8 of the 24-byte nonce
    chacha_nonce = b"\x00\x00\x00\x00" + nonce24[16:]
    return ChaCha20Poly1305(subkey).encrypt(chacha_nonce, plaintext, aad)


def _xchacha20poly1305_decrypt(
    key: bytes, nonce24: bytes, ciphertext: bytes, aad: bytes
) -> bytes:
    if len(nonce24) != 24:
        raise ValueError("XChaCha20-Poly1305 nonce must be 24 bytes")
    subkey = _hchacha20(key, nonce24[:16])
    chacha_nonce = b"\x00\x00\x00\x00" + nonce24[16:]
    return ChaCha20Poly1305(subkey).decrypt(chacha_nonce, ciphertext, aad)


def encrypt_ncryptsec(
    secret: bytes,
    password: str,
    *,
    log_n: int = 16,
    key_security: int = 0x02,
) -> str:
    """Encrypt a 32-byte secp256k1 secret → ncryptsec1…"""
    if len(secret) != 32:
        raise ValueError("secret must be 32 bytes")
    if not password:
        raise ValueError("password is required")
    if key_security not in (0x00, 0x01, 0x02):
        raise ValueError("key_security must be 0x00, 0x01, or 0x02")
    salt = os.urandom(16)
    key = _scrypt_key(password, salt, log_n)
    nonce = os.urandom(24)
    aad = bytes([key_security])
    ciphertext = _xchacha20poly1305_encrypt(key, nonce, secret, aad)
    payload = bytes([_VERSION, log_n]) + salt + nonce + aad + ciphertext
    if len(payload) != _PAYLOAD_LEN:
        raise ValueError(f"unexpected payload length {len(payload)}")
    return bech32_encode("ncryptsec", payload)


def decrypt_ncryptsec(ncryptsec: str, password: str) -> bytes:
    """Decrypt ncryptsec1… → 32-byte secret."""
    if not password:
        raise ValueError("password is required")
    hrp, data = bech32_decode(ncryptsec.strip())
    if hrp != "ncryptsec":
        raise ValueError("invalid ncryptsec prefix")
    if len(data) != _PAYLOAD_LEN:
        raise ValueError("invalid ncryptsec payload length")
    if data[0] != _VERSION:
        raise ValueError(f"unsupported ncryptsec version {data[0]}")
    log_n = data[1]
    salt = data[2:18]
    nonce = data[18:42]
    aad = data[42:43]
    ciphertext = data[43:]
    key = _scrypt_key(password, salt, log_n)
    try:
        secret = _xchacha20poly1305_decrypt(key, nonce, ciphertext, aad)
    except Exception as exc:
        raise ValueError("wrong passphrase or corrupted ncryptsec") from exc
    if len(secret) != 32:
        raise ValueError("decrypted secret has invalid length")
    return secret
