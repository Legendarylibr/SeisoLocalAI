"""Endpoint resolver: loopback only, never_leave blocks router."""

from __future__ import annotations

from seiso.agent.adapters.endpoint import resolve_endpoint
from seiso.agent.policy import RouteClass


def test_ollama_source_no_probe() -> None:
    endpoint = resolve_endpoint(source="ollama", probe=False)
    assert endpoint.source == "ollama"
    assert "127.0.0.1" in endpoint.url
    public = endpoint.public_dict()
    assert "api_key" not in public
    assert public["host"]


def test_router_never_leave(monkeypatch) -> None:
    monkeypatch.setenv("SEISO_MODEL_ROUTER_ENABLED", "true")
    monkeypatch.setenv("SEISO_MODEL_ROUTER_URL", "http://127.0.0.1:8780")
    endpoint = resolve_endpoint(source="router", route_class=RouteClass.NEVER_LEAVE, probe=False)
    assert endpoint.source == "none"
    assert endpoint.reason == "router_unavailable"


def test_router_rejects_remote(monkeypatch) -> None:
    monkeypatch.setenv("SEISO_MODEL_ROUTER_ENABLED", "true")
    monkeypatch.setenv("SEISO_MODEL_ROUTER_URL", "https://evil.example/v1")
    endpoint = resolve_endpoint(source="router", probe=False)
    assert endpoint.source == "none"


def test_auto_without_probe_picks_something_local() -> None:
    endpoint = resolve_endpoint(source="auto", probe=False)
    assert endpoint.source in {"ollama", "router", "forge"}
    assert endpoint.url
