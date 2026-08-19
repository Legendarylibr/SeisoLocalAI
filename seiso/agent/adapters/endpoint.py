"""Resolve a localhost OpenAI-compat URL for child harnesses."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from seiso.agent.policy import RouteClass, parse_route_class
from seiso.routing.external import is_loopback_url, validate_router_url

OLLAMA_DEFAULT = "http://127.0.0.1:11434/v1"
FORGE_DEFAULT = "http://127.0.0.1:8765/v1"
_PROBE_TIMEOUT = 1.5


@dataclass(frozen=True, slots=True)
class ResolvedEndpoint:
    url: str
    source: str
    model_id: str
    api_key: str = ""
    reason: str = ""

    def public_dict(self) -> dict[str, Any]:
        parsed = urlparse(self.url)
        host = parsed.netloc or parsed.path
        return {
            "url": self.url,
            "host": host,
            "source": self.source,
            "model_id": self.model_id,
            "reason": self.reason,
            "has_key": bool(self.api_key),
        }


def _probe(url: str) -> bool:
    raw = (url or "").strip()
    if not raw:
        return False
    try:
        with urlopen(raw, timeout=_PROBE_TIMEOUT) as resp:  # noqa: S310
            return 200 <= int(getattr(resp, "status", 200)) < 500
    except Exception:
        return False


def _inference_api_key(data_dir: Path | None) -> str:
    env = (os.environ.get("SEISO_INFERENCE_API_KEY") or "").strip()
    if env:
        return env
    if data_dir is None:
        return ""
    path = Path(data_dir) / ".inference_api_key"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _router_url() -> str | None:
    enabled = (os.environ.get("SEISO_MODEL_ROUTER_ENABLED") or "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None
    raw = (os.environ.get("SEISO_MODEL_ROUTER_URL") or "http://127.0.0.1:8780").strip()
    try:
        return validate_router_url(raw)
    except ValueError:
        return None


def resolve_endpoint(
    *,
    source: str = "auto",
    model_id: str = "default",
    data_dir: Path | None = None,
    route_class: str | RouteClass | None = None,
    forge_url: str | None = None,
    probe: bool = True,
) -> ResolvedEndpoint:
    """Pick one OpenAI-compat URL. Prefer Ollama / router over Forge ``/v1``.

    ``source``: auto | ollama | router | forge | local
    """
    policy = parse_route_class(route_class)
    wanted = (source or "auto").strip().lower()
    if wanted == "local":
        wanted = "auto"
    model = (model_id or "default").strip() or "default"

    def _ollama() -> ResolvedEndpoint | None:
        url = (os.environ.get("SEISO_OLLAMA_URL") or OLLAMA_DEFAULT).rstrip("/")
        if not url.endswith("/v1"):
            url = url.rstrip("/") + "/v1"
        if not is_loopback_url(url):
            return None
        tags = url[: -len("/v1")] + "/api/tags" if url.endswith("/v1") else url
        if probe and not (_probe(tags) or _probe(url + "/models")):
            return None
        return ResolvedEndpoint(url, "ollama", model, reason="ollama_healthy")

    def _router() -> ResolvedEndpoint | None:
        if policy is RouteClass.NEVER_LEAVE:
            return None
        base = _router_url()
        if not base:
            return None
        chat = base if base.endswith("/v1") else base.rstrip("/") + "/v1"
        if probe and not _probe(base + "/health"):
            return None
        return ResolvedEndpoint(chat, "router", "__seiso_router__", reason="external_router")

    def _forge() -> ResolvedEndpoint | None:
        raw = (forge_url or os.environ.get("SEISO_FORGE_V1") or FORGE_DEFAULT).rstrip("/")
        if not raw.endswith("/v1"):
            raw = raw + "/v1"
        if not is_loopback_url(raw):
            return None
        key = _inference_api_key(data_dir)
        health = raw[: -len("/v1")] + "/health"
        if probe and not _probe(health):
            return None
        return ResolvedEndpoint(raw, "forge", model, api_key=key, reason="forge_v1")

    if wanted == "ollama":
        hit = _ollama()
        return hit or ResolvedEndpoint("", "none", model, reason="ollama_unavailable")
    if wanted == "router":
        hit = _router()
        return hit or ResolvedEndpoint("", "none", model, reason="router_unavailable")
    if wanted == "forge":
        hit = _forge()
        return hit or ResolvedEndpoint("", "none", model, reason="forge_unavailable")

    for factory in (_ollama, _router, _forge):
        hit = factory()
        if hit is not None:
            return hit
    return ResolvedEndpoint(
        "",
        "none",
        model,
        reason="no_local_endpoint",
    )
