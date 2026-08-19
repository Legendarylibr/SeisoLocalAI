"""Generic signed-relay agent interactions (Buzz authority)."""

from __future__ import annotations

import pytest

from seiso.research.nostr.events import verify_event
from seiso.research.nostr.keys import generate_keypair


def test_signed_agent_interaction_relay_only_with_signing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = generate_keypair()
    monkeypatch.setenv("BUZZ_PRIVATE_KEY", pair.nsec)
    from seiso.agent.signed_relay import signed_agent_interaction

    out = signed_agent_interaction(
        role="train",
        status="started",
        channel="ch-1",
        job_id="job-1",
        message="smoke",
    )
    assert out["buzz_receipt"]["relay_policy"] == "signed_event_only"
    assert out["buzz_receipt"]["npub"] == pair.npub
    assert out["nostr_event"]["kind"] == 31254
    assert verify_event(out["nostr_event"])
    d_tags = [t[1] for t in out["nostr_event"]["tags"] if t and t[0] == "d"]
    assert d_tags == ["job-1:train:started"]
    assert "Relay policy" in out["note"]
    assert "nsec" not in str(out["buzz_receipt"]).lower()


def test_receipt_allows_tokenizer_field() -> None:
    from seiso.agent.receipts import agent_receipt

    receipt = agent_receipt(
        role="train",
        status="started",
        tokenizer="llama",
        token_count=12,
        mesh_token="should-drop",
    )
    assert receipt["tokenizer"] == "llama"
    assert receipt["token_count"] == 12
    assert "mesh_token" not in receipt


def test_signed_agent_interaction_local_unsigned_without_nsec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BUZZ_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("BUZZ_AUTH_TAG", raising=False)
    from seiso.agent.signed_relay import signed_agent_interaction

    out = signed_agent_interaction(
        role="doctor",
        status="ok",
        require_nsec=False,
    )
    assert out["nostr_event"] is None
    assert out["buzz_receipt"]["relay_policy"] == "local_unsigned_not_authority"


def test_relay_signed_event_refuses_garbage() -> None:
    from seiso.agent.signed_relay import relay_signed_event

    with pytest.raises(RuntimeError, match="Relay only with signing"):
        relay_signed_event(None)
    with pytest.raises(RuntimeError, match="Relay only with signing"):
        relay_signed_event({"id": "ab" * 32, "sig": "00" * 64})
