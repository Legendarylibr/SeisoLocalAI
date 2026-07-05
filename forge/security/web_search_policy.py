"""Policy for agent web search — query bounds, HTTPS-only results, public hosts only."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from forge.security.code_policy import normalize_user_text, scrub_secrets
from forge.security.url_policy import _is_blocked_ip, _resolve_host
from seiso.security import SecurityError

_MAX_QUERY_LEN = 240
_MAX_SNIPPET_LEN = 600
_MAX_RESULTS = 8

_BLOCKED_QUERY = re.compile(
    r"(?i)\b(site:|file:|localhost|127\.0\.0\.1|0\.0\.0\.0|169\.254\.169\.254|metadata\.google)\b"
)


def normalize_search_query(query: str) -> str:
    """Normalize and validate a user/agent search query."""
    cleaned = normalize_user_text(query, max_len=_MAX_QUERY_LEN)
    if not cleaned:
        raise ValueError("Search query is required")
    if _BLOCKED_QUERY.search(cleaned):
        raise SecurityError("Search query contains blocked patterns")
    return cleaned


def sanitize_result_snippet(text: str) -> str:
    snippet = scrub_secrets(normalize_user_text(text, max_len=_MAX_SNIPPET_LEN))
    return snippet.strip()


def validate_public_https_url(url: str) -> str | None:
    """Return normalized HTTPS URL if public, else None."""
    raw = (url or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme.lower() != "https":
        return None
    if parsed.username or parsed.password:
        return None
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        return None
    try:
        addrs = _resolve_host(host)
    except SecurityError:
        return None
    if any(_is_blocked_ip(addr) for addr in addrs):
        return None
    port = parsed.port
    netloc = host if port is None else f"{host}:{port}"
    path = parsed.path or ""
    return f"https://{netloc}{path}"
