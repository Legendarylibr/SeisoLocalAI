"""Nostr relay URL policy (SSRF) and NIP-01 publish/fetch protocol (mocked WS)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from seiso.research.nostr.events import build_attestation_event
from seiso.research.nostr.keys import generate_keypair
from seiso.research.nostr.policy import (
    normalize_relay_list,
    nostr_allowed,
    nostr_auto_attest_enabled,
    relay_allowlist_from_env,
    validate_relay_url,
)
from seiso.research.nostr.relays import fetch_event_by_id, publish_event
from seiso.security import SecurityError


@pytest.mark.parametrize(
    "url,match",
    [
        ("", "required"),
        ("not-a-url", "scheme|host|invalid"),
        ("https://relay.example.com", "scheme"),
        ("wss://user:pass@relay.example.com", "credentials"),
        ("wss://169.254.169.254", "not allowed|blocked"),
        ("wss://metadata.google.internal", "not allowed"),
        ("wss://10.0.0.1", "not allowed|blocked|private"),
        ("wss://192.168.0.5", "not allowed|blocked|private"),
        ("wss://100.64.0.1", "not allowed|blocked"),
        ("wss://[::1]", "not allowed|loopback|blocked"),
        ("ws://example.com", "loopback|ws"),
        ("wss://127.0.0.1", "not allowed|loopback|blocked"),
    ],
)
def test_validate_relay_url_rejects_dangerous(url: str, match: str):
    with pytest.raises(SecurityError, match=match):
        validate_relay_url(url)


def test_validate_relay_url_blocks_cgnat_shared_address_space():
    """RFC 6598 100.64.0.0/10 is not ipaddress.is_private but must be blocked."""
    with pytest.raises(SecurityError, match="not allowed|blocked"):
        validate_relay_url("wss://100.64.12.34")


@pytest.fixture
def public_dns(monkeypatch):
    def fake_getaddrinfo(host, *args, **kwargs):
        return [(None, None, None, None, ("8.8.8.8", 0))]

    monkeypatch.setattr("seiso.research.nostr.policy.socket.getaddrinfo", fake_getaddrinfo)


def test_validate_relay_url_allowlist_and_loopback(public_dns):
    ok = validate_relay_url(
        "wss://relay.example.com/path/",
        allowlist=["relay.example.com"],
    )
    assert ok == "wss://relay.example.com/path"
    with pytest.raises(SecurityError, match="allowlist"):
        validate_relay_url(
            "wss://relay.example.com",
            allowlist=["other.example.com"],
        )
    loop = validate_relay_url("ws://127.0.0.1:10547", allow_loopback=True)
    assert loop == "ws://127.0.0.1:10547"
    with pytest.raises(SecurityError, match="loopback"):
        validate_relay_url("wss://localhost", allow_loopback=False)


def test_validate_relay_url_blocks_dns_to_private(monkeypatch):
    def fake_getaddrinfo(host, *args, **kwargs):
        return [(None, None, None, None, ("10.1.2.3", 0))]

    monkeypatch.setattr("seiso.research.nostr.policy.socket.getaddrinfo", fake_getaddrinfo)
    with pytest.raises(SecurityError, match="blocked"):
        validate_relay_url("wss://evil.example.com")


def test_normalize_relay_list_dedupes_and_requires_one(public_dns):
    with pytest.raises(SecurityError, match="at least one"):
        normalize_relay_list([])
    urls = normalize_relay_list(
        [
            "wss://relay.example.com/",
            "wss://relay.example.com",
            "ws://127.0.0.1:9",
        ],
        allowlist=["relay.example.com", "127.0.0.1"],
        allow_loopback=True,
    )
    assert urls == ["wss://relay.example.com", "ws://127.0.0.1:9"]


def test_env_gates(monkeypatch):
    monkeypatch.delenv("SEISO_ALLOW_NOSTR", raising=False)
    monkeypatch.delenv("SEISO_NOSTR_ATTEST", raising=False)
    monkeypatch.delenv("SEISO_NOSTR_RELAYS", raising=False)
    assert nostr_allowed() is True
    assert nostr_auto_attest_enabled() is False
    assert relay_allowlist_from_env() == ["wss://nos.lol", "wss://relay.damus.io"]

    monkeypatch.setenv("SEISO_ALLOW_NOSTR", "0")
    assert nostr_allowed() is False
    assert nostr_auto_attest_enabled() is False
    assert relay_allowlist_from_env() == []

    monkeypatch.setenv("SEISO_ALLOW_NOSTR", "1")
    monkeypatch.setenv("SEISO_NOSTR_ATTEST", "true")
    monkeypatch.setenv("SEISO_NOSTR_RELAYS", "wss://a.example, wss://b.example")
    assert nostr_auto_attest_enabled() is True
    assert relay_allowlist_from_env() == ["wss://a.example", "wss://b.example"]


class _FakeWS:
    def __init__(self, responses: list[Any]):
        self._responses = list(responses)
        self.sent: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def send(self, data: str):
        self.sent.append(data)

    def recv(self, timeout: float = 0):
        if not self._responses:
            raise TimeoutError("no more responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, str) else json.dumps(item)


def test_publish_event_requires_ok_ack():
    pair = generate_keypair()
    event = build_attestation_event(
        pair=pair,
        attestation_json='{"schema":"seiso.provenance.attestation/v1"}',
        pipeline="compress",
        run_id="r1",
        created_at=1,
    )

    ok_ws = _FakeWS(
        [
            ["NOTICE", "hi"],
            ["OK", event["id"], True, ""],
        ]
    )
    with patch(
        "seiso.research.nostr.relays._require_websockets", return_value=lambda *a, **k: ok_ws
    ):
        with patch(
            "seiso.research.nostr.relays.validate_relay_url",
            side_effect=lambda u, **kw: u,
        ):
            accepted = publish_event(event, ["wss://relay.example.com"])
    assert accepted == ["wss://relay.example.com"]
    assert json.loads(ok_ws.sent[0]) == ["EVENT", event]

    fail_ws = _FakeWS([["OK", event["id"], False, "pow"]])
    with patch(
        "seiso.research.nostr.relays._require_websockets", return_value=lambda *a, **k: fail_ws
    ):
        with patch(
            "seiso.research.nostr.relays.validate_relay_url",
            side_effect=lambda u, **kw: u,
        ):
            with pytest.raises(SecurityError, match="no allowlisted relay accepted"):
                publish_event(event, ["wss://relay.example.com"])


def test_fetch_event_by_id_returns_matching_and_handles_eose():
    event = {"id": "ab" * 32, "pubkey": "cd" * 32, "kind": 31250, "content": "{}"}
    # Capture sub_id from first send and craft matching EVENT.
    real_event = dict(event)

    class _CaptureWS(_FakeWS):
        def send(self, data: str):
            super().send(data)
            msg = json.loads(data)
            if msg[0] == "REQ":
                sub = msg[1]
                self._responses = [
                    ["EVENT", sub, real_event],
                    ["EOSE", sub],
                ]

    cap = _CaptureWS([])
    with patch("seiso.research.nostr.relays._require_websockets", return_value=lambda *a, **k: cap):
        with patch(
            "seiso.research.nostr.relays.validate_relay_url",
            side_effect=lambda u, **kw: u,
        ):
            got = fetch_event_by_id("AB" * 32, ["wss://relay.example.com"])
    assert got == real_event

    with pytest.raises(ValueError, match="64-char"):
        fetch_event_by_id("short", ["wss://relay.example.com"])


def test_fetch_event_missing_returns_none():
    class _EoseWS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def send(self, data: str):
            self._sub = json.loads(data)[1]

        def recv(self, timeout: float = 0):
            return json.dumps(["EOSE", self._sub])

    with patch(
        "seiso.research.nostr.relays._require_websockets", return_value=lambda *a, **k: _EoseWS()
    ):
        with patch(
            "seiso.research.nostr.relays.validate_relay_url",
            side_effect=lambda u, **kw: u,
        ):
            assert fetch_event_by_id("ab" * 32, ["wss://relay.example.com"]) is None


def test_fetch_handles_closed_and_addressable_filter():
    from seiso.research.nostr.relays import fetch_addressable_event

    class _ClosedWS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def send(self, data: str):
            self._sub = json.loads(data)[1]

        def recv(self, timeout: float = 0):
            return json.dumps(["CLOSED", self._sub, "error: idle"])

    with patch(
        "seiso.research.nostr.relays._require_websockets",
        return_value=lambda *a, **k: _ClosedWS(),
    ):
        with patch(
            "seiso.research.nostr.relays.validate_relay_url",
            side_effect=lambda u, **kw: u,
        ):
            assert fetch_event_by_id("ab" * 32, ["wss://relay.example.com"]) is None

    event = {
        "id": "ab" * 32,
        "pubkey": "cd" * 32,
        "kind": 31250,
        "tags": [["d", "compress:run1"]],
        "content": "{}",
    }

    class _AddrWS(_FakeWS):
        def send(self, data: str):
            super().send(data)
            msg = json.loads(data)
            assert msg[0] == "REQ"
            filt = msg[2]
            assert filt["authors"] == ["cd" * 32]
            assert filt["kinds"] == [31250]
            assert filt["#d"] == ["compress:run1"]
            self._responses = [["EVENT", msg[1], event], ["EOSE", msg[1]]]

    addr = _AddrWS([])
    with patch(
        "seiso.research.nostr.relays._require_websockets",
        return_value=lambda *a, **k: addr,
    ):
        with patch(
            "seiso.research.nostr.relays.validate_relay_url",
            side_effect=lambda u, **kw: u,
        ):
            got = fetch_addressable_event(
                pubkey="CD" * 32,
                kind=31250,
                d_tag="compress:run1",
                relays=["wss://relay.example.com"],
            )
    assert got == event


def test_publish_surfaces_missing_websockets_deps():
    with patch(
        "seiso.research.nostr.relays._require_websockets",
        side_effect=ImportError(
            "Nostr relay I/O requires optional deps: pip install 'seiso[nostr]'"
        ),
    ):
        with pytest.raises(ImportError, match="seiso\\[nostr\\]"):
            publish_event({"id": "ab" * 32}, ["wss://relay.example.com"])
