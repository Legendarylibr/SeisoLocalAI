"""Sandboxed web search — disabled in local-only mode (no external APIs)."""

from __future__ import annotations

import json


def web_search(query: str, max_results: int = 5) -> str:
    """Return a local-only stub; external search is disabled."""
    _ = max_results
    query = (query or "").strip()
    return json.dumps(
        {
            "error": "Web search is disabled in local-only mode",
            "query": query,
            "hint": "Remove the web_search tool or enable external integrations when supported.",
        }
    )
