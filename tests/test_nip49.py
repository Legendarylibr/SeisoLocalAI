"""NIP-49 ncryptsec encrypt/decrypt."""

from __future__ import annotations

import unicodedata

import pytest

from seiso.research.nostr.bech32 import bech32_decode, bech32_encode
from seiso.research.nostr.nip49 import decrypt_ncryptsec, encrypt_ncryptsec

# Official NIP-49 decryption vector (test-only; not production secrets).
_NIP49_TEST_NCRYPTSEC = (
    "ncryptsec1qgg9947rlpvqu76pj5ecreduf9jxhselq2nae2kghhvd5g7dgjtcxfqtd67p9m0w57l"
    "spw8gsq6yphnm8623nsl8xn9j4jdzz84zm3frztj3z7s35vpzmqf6ksu8r89qk5z2zxfmu5gv8th8wclt0h4p"
)
_NIP49_TEST_PASSWORD = "nostr"
_NIP49_TEST_SECRET_HEX = (
    "3501454135014541350145413501453fefb02227e449e57cf4d3a3ce05378683"
)


def _mutate_log_n(ncryptsec: str, log_n: int) -> str:
    hrp, data = bech32_decode(ncryptsec.strip())
    assert hrp == "ncryptsec"
    mutated = bytearray(data)
    mutated[1] = log_n & 0xFF
    return bech32_encode("ncryptsec", bytes(mutated))


def test_nip49_official_decrypt_vector():
    secret = decrypt_ncryptsec(_NIP49_TEST_NCRYPTSEC, _NIP49_TEST_PASSWORD)
    assert secret.hex() == _NIP49_TEST_SECRET_HEX


def test_nip49_roundtrip():
    secret = bytes.fromhex(_NIP49_TEST_SECRET_HEX)
    enc = encrypt_ncryptsec(secret, "unit-test-pass", log_n=16, key_security=0x00)
    assert enc.startswith("ncryptsec1")
    assert decrypt_ncryptsec(enc, "unit-test-pass") == secret


def test_nip49_wrong_password():
    with pytest.raises(ValueError, match="wrong passphrase"):
        decrypt_ncryptsec(_NIP49_TEST_NCRYPTSEC, "not-the-password")


def test_nip49_password_nfkc_normalization():
    secret = bytes.fromhex(_NIP49_TEST_SECRET_HEX)
    password = "ÅΩẛ̣"  # U+212B U+2126 U+1E9B U+0323
    enc = encrypt_ncryptsec(secret, password, log_n=16)
    normalized = unicodedata.normalize("NFKC", password)
    assert decrypt_ncryptsec(enc, normalized) == secret


def test_nip49_decrypt_rejects_log_n_out_of_range():
    secret = bytes.fromhex(_NIP49_TEST_SECRET_HEX)
    enc = encrypt_ncryptsec(secret, "bound-check", log_n=16)
    # Cap is 18 (stricter than NIP-49's 22) to avoid scrypt DoS on decrypt.
    for bad in (0, 19, 22, 23, 30, 255):
        mutated = _mutate_log_n(enc, bad)
        with pytest.raises(ValueError, match="log_n"):
            decrypt_ncryptsec(mutated, "bound-check")


def test_nip49_encrypt_rejects_log_n_out_of_range():
    secret = bytes.fromhex(_NIP49_TEST_SECRET_HEX)
    with pytest.raises(ValueError, match="log_n"):
        encrypt_ncryptsec(secret, "x", log_n=0)
    with pytest.raises(ValueError, match="log_n"):
        encrypt_ncryptsec(secret, "x", log_n=19)
    with pytest.raises(ValueError, match="log_n"):
        encrypt_ncryptsec(secret, "x", log_n=23)


def test_nip49_rejects_empty_and_whitespace_password():
    secret = bytes.fromhex(_NIP49_TEST_SECRET_HEX)
    for bad in ("", "   ", "\t"):
        with pytest.raises(ValueError, match="password"):
            encrypt_ncryptsec(secret, bad)
        with pytest.raises(ValueError, match="password"):
            decrypt_ncryptsec(_NIP49_TEST_NCRYPTSEC, bad)


def test_nip49_rejects_bad_secret_length_and_key_security():
    with pytest.raises(ValueError, match="32 bytes"):
        encrypt_ncryptsec(b"short", "pass")
    secret = bytes.fromhex(_NIP49_TEST_SECRET_HEX)
    with pytest.raises(ValueError, match="key_security"):
        encrypt_ncryptsec(secret, "pass", key_security=0x03)


def test_nip49_rejects_invalid_prefix_and_payload():
    with pytest.raises(ValueError, match="prefix|ncryptsec|bech32|checksum"):
        decrypt_ncryptsec("nsec1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq", "x")
    # Valid bech32 checksum with wrong HRP.
    wrong_hrp = bech32_encode("npub", bytes(32))
    with pytest.raises(ValueError, match="prefix"):
        decrypt_ncryptsec(wrong_hrp, "x")
