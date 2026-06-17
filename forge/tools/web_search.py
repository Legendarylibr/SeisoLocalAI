"""Sandboxed web search — DuckDuckGo lite HTML, no API key required."""

from __future__ import annotations

import json
import re
from urllib.parse import urlparse

import httpx

from forge.tools.sanitize import wrap_tool_result

_USER_AGENT = "Seiso-Forge/0.1 (local-first; +https://github.com/Legendarylibr/SeisoLocalAI)"
_MAX_QUERY_LEN = 512
_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})


def _sanitize_result_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    if parsed.scheme.lower() not in _ALLOWED_URL_SCHEMES:
        return ""
    if parsed.username or parsed.password:
        return ""
    host = (parsed.hostname or "").lower()
    if not host or host in {"localhost", "127.0.0.1", "::1"}:
        return ""
    return raw[:2048]


def web_search(query: str, max_results: int = 5) -> str:
    query = (query or "").strip()[:_MAX_QUERY_LEN]
    if not query:
        return json.dumps({"error": "Empty search query", "query": query})

    url = "https://lite.duckduckgo.com/lite/"
    try:
        with httpx.Client(timeout=15.0, follow_redirects=False) as client:
            resp = client.post(
                url,
                data={"q": query},
                headers={"User-Agent": _USER_AGENT},
            )
            resp.raise_for_status()
            html = resp.text
    except httpx.HTTPError as exc:
        return json.dumps({"error": f"Search failed: {exc}", "query": query})

    results = _parse_lite_results(html, max_results)
    payload = {"query": query, "results": results}
    return wrap_tool_result("web_search", json.dumps(payload))


def _parse_lite_results(html: str, max_results: int) -> list[dict]:
    """Extract result snippets from DuckDuckGo lite HTML."""
    results: list[dict] = []
    link_pattern = re.compile(
        r'<a rel="nofollow" href="([^"]+)"[^>]*>([^<]+)</a>',
        re.IGNORECASE,
    )
    snippet_pattern = re.compile(
        r'<td class="result-snippet"[^>]*>([^<]+(?:<[^>]+>[^<]*)*)</td>',
        re.IGNORECASE,
    )
    links = link_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    for i, (href, title) in enumerate(links[: max_results * 2]):
        safe_href = _sanitize_result_url(href)
        if not safe_href:
            continue
        snippet = snippets[i] if i < len(snippets) else ""
        snippet = re.sub(r"<[^>]+>", "", snippet).strip()
        results.append({"title": title.strip()[:200], "url": safe_href, "snippet": snippet[:500]})
        if len(results) >= max_results:
            break

    if not results:
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()[:800]
        if text:
            results.append({"title": "Raw", "url": "", "snippet": text})

    return results
