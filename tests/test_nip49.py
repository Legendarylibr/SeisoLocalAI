"""NIP-49 ncryptsec encrypt/decrypt."""

from __future__ import annotations

import pytest

from seiso.research.nostr.nip49 import (
    NIP49_TEST_NCRYPTSEC,
    NIP49_TEST_PASSWORD,
    NIP49_TEST_SECRET_HEX,
    decrypt_ncryptsec,
    encrypt_ncryptsec,
)


def test_nip49_official_decrypt_vector():
    secret = decrypt_ncryptsec(NIP49_TEST_NCRYPTSEC, NIP49_TEST_PASSWORD)
    assert secret.hex() == NIP49_TEST_SECRET_HEX


def test_nip49_roundtrip():
    secret = bytes.fromhex(NIP49_TEST_SECRET_HEX)
    enc = encrypt_ncryptsec(secret, "unit-test-pass", log_n=16, key_security=0x00)
    assert enc.startswith("ncryptsec1")
    assert decrypt_ncryptsec(enc, "unit-test-pass") == secret


def test_nip49_wrong_password():
    with pytest.raises(ValueError, match="wrong passphrase"):
        decrypt_ncryptsec(NIP49_TEST_NCRYPTSEC, "not-the-password")


def test_nip49_password_nfkc_normalization():
    # ÅΩṡ (precomposed / compatibility) — NFKC of the NIP example family.
    secret = bytes.fromhex(NIP49_TEST_SECRET_HEX)
    # Use a password that changes under NFKC
    password = "ÅΩẛ̣"  # U+212B U+2126 U+1E9B U+0323
    enc = encrypt_ncryptsec(secret, password, log_n=16)
    # Same visual after NFKC should decrypt
    import unicodedata

    normalized = unicodedata.normalize("NFKC", password)
    assert decrypt_ncryptsec(enc, normalized) == secret
