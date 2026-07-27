"""NIP-49 ncryptsec encrypt/decrypt."""

from __future__ import annotations

import unicodedata

import pytest

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
    # ÅΩṡ (precomposed / compatibility) — NFKC of the NIP example family.
    secret = bytes.fromhex(_NIP49_TEST_SECRET_HEX)
    # Use a password that changes under NFKC
    password = "ÅΩẛ̣"  # U+212B U+2126 U+1E9B U+0323
    enc = encrypt_ncryptsec(secret, password, log_n=16)
    # Same visual after NFKC should decrypt
    normalized = unicodedata.normalize("NFKC", password)
    assert decrypt_ncryptsec(enc, normalized) == secret
