"""Opt-in Nostr attestation for Seiso research provenance (digests only)."""

from __future__ import annotations

from seiso.research.nostr.attest import (
    attest_manifest,
    maybe_auto_attest,
    verify_attestation,
)
from seiso.research.nostr.policy import nostr_allowed, nostr_auto_attest_enabled

__all__ = [
    "attest_manifest",
    "maybe_auto_attest",
    "nostr_allowed",
    "nostr_auto_attest_enabled",
    "verify_attestation",
]
