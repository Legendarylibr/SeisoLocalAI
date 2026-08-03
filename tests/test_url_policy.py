"""URL / SSRF policy and pinned-endpoint hardening."""

from __future__ import annotations

import pytest

from forge.security.http_client import pin_request_to_ip
from forge.security.url_policy import (
    resolve_pinned_endpoint,
    validate_provider_base_url,
)
from seiso.security import SecurityError

def test_provider_url_blocks_metadata_ip():
    with pytest.raises(SecurityError):
        validate_provider_base_url("http://169.254.169.254/latest/meta-data/", provider_type="vllm")

def test_provider_url_blocks_cgnat_shared_address_space():
    """RFC 6598 100.64.0.0/10 is not ipaddress.is_private but must be blocked."""
    with pytest.raises(SecurityError):
        validate_provider_base_url(
            "https://100.64.0.1:443/v1", provider_type="remote_chat"
        )

def test_provider_url_blocks_http_non_local():
    with pytest.raises(SecurityError):
        validate_provider_base_url("http://example.com/v1", provider_type="vllm")

def test_provider_url_fails_on_unresolvable_host():
    with pytest.raises(SecurityError, match="could not be resolved"):
        validate_provider_base_url(
            "https://this-host-definitely-does-not-exist-xyz123.invalid/v1",
            provider_type="remote_chat",
        )

def test_local_chat_rejects_public_https():
    """local_chat must not accept arbitrary public HTTPS (cloud gate bypass)."""
    with pytest.raises(SecurityError, match="loopback"):
        validate_provider_base_url(
            "https://example.com/v1", provider_type="local_chat"
        )
    with pytest.raises(SecurityError, match="loopback"):
        validate_provider_base_url("https://example.com/v1", provider_type="vllm")

def test_resolve_pinned_endpoint_pins_remote_host(monkeypatch):
    monkeypatch.setattr(
        "forge.security.url_policy._resolve_host",
        lambda host: ["93.184.216.34"],
    )
    endpoint = resolve_pinned_endpoint(
        "https://example.com/v1", provider_type="remote_chat"
    )
    assert endpoint.pinned_ip == "93.184.216.34"
    assert endpoint.host == "example.com"
    assert endpoint.base_url == "https://example.com/v1"

def test_resolve_pinned_endpoint_pins_loopback_dns_names(monkeypatch):
    """DNS names that currently resolve to loopback must still be pinned."""
    monkeypatch.setattr(
        "forge.security.url_policy._resolve_host",
        lambda host: ["127.0.0.1"],
    )
    endpoint = resolve_pinned_endpoint(
        "http://local-vllm.test:8000", provider_type="local_chat"
    )
    assert endpoint.pinned_ip == "127.0.0.1"
    assert endpoint.host == "local-vllm.test"

def test_pin_request_to_ip_preserves_host_and_sni_without_getaddrinfo_patch():
    import socket

    import httpx

    real_getaddrinfo = socket.getaddrinfo
    request = httpx.Request("POST", "https://example.com/v1/chat/completions", json={})
    pinned = pin_request_to_ip(
        request, host="example.com", pinned_ip="93.184.216.34"
    )

    assert pinned.url.host == "93.184.216.34"
    assert pinned.headers.get("host") == "example.com"
    assert pinned.extensions.get("sni_hostname") == "example.com"
    assert socket.getaddrinfo is real_getaddrinfo

def test_pin_request_to_ip_ignores_unrelated_hosts():
    import httpx

    request = httpx.Request("GET", "https://other.example/v1")
    pinned = pin_request_to_ip(
        request, host="example.com", pinned_ip="93.184.216.34"
    )
    assert pinned is request
    assert pinned.url.host == "other.example"

def test_provider_url_blocks_embedded_credentials():
    with pytest.raises(SecurityError, match="credentials"):
        validate_provider_base_url("https://user:pass@example.com/v1", provider_type="vllm")

def test_provider_url_blocks_decimal_metadata_ip():
    with pytest.raises(SecurityError):
        validate_provider_base_url("https://2852039166/", provider_type="vllm")

def test_provider_url_blocks_ipv6_mapped_metadata():
    with pytest.raises(SecurityError):
        validate_provider_base_url("https://[::ffff:169.254.169.254]/", provider_type="vllm")

def test_provider_url_allows_shorthand_loopback_for_local_vllm_http():
    url = validate_provider_base_url("http://127.1:8000/", provider_type="vllm")
    assert url.startswith("http://127.1:8000")

def test_provider_url_allows_shorthand_loopback_for_local_vllm():
    url = validate_provider_base_url("https://127.1:8000/", provider_type="vllm")
    assert url.startswith("https://127.1:8000")

@pytest.mark.asyncio
async def test_provider_ssrf_blocked_on_create(app, auth_client):
    client, _token, headers, _tmp = auth_client
    res = await client.post(
        "/api/providers",
        headers=headers,
        json={
            "name": "Evil",
            "provider_type": "vllm",
            "config": {"base_url": "http://169.254.169.254/"},
        },
    )
    assert res.status_code == 400

