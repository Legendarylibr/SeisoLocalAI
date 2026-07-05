"""Sandboxed web search — delegates to secure service when enabled."""

from __future__ import annotations

import json

from forge.config import get_settings
from forge.services.web_search import secure_web_search_sync, web_search_available


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web when SEISO_WEB_SEARCH_ENABLED=1; otherwise return a stub."""
    settings = get_settings()
    if not web_search_available(settings):
        query = (query or "").strip()
        return json.dumps(
            {
                "error": "Web search is disabled on this Forge server",
                "query": query,
                "hint": "Set SEISO_WEB_SEARCH_ENABLED=1 (optional: SEISO_BRAVE_SEARCH_API_KEY for Brave).",
            }
        )
    return secure_web_search_sync(query, settings, max_results=max_results)
