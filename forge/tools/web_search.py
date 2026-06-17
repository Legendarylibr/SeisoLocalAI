"""Sandboxed web search — DuckDuckGo lite HTML, no API key required."""

from __future__ import annotations

import json
import re

import httpx

_USER_AGENT = "Seiso-Forge/0.1 (local-first; +https://github.com/seiso-ai/seiso)"


def web_search(query: str, max_results: int = 5) -> str:
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
    return json.dumps({"query": query, "results": results})


def _parse_lite_results(html: str, max_results: int) -> list[dict]:
    """Extract result snippets from DuckDuckGo lite HTML."""
    results: list[dict] = []
    # Links in lite results: <a rel="nofollow" href="...">title</a>
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

    for i, (href, title) in enumerate(links[:max_results]):
        snippet = snippets[i] if i < len(snippets) else ""
        snippet = re.sub(r"<[^>]+>", "", snippet).strip()
        results.append({"title": title.strip(), "url": href, "snippet": snippet[:500]})

    if not results:
        # Fallback: strip tags and return truncated text
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()[:800]
        results.append({"title": "Raw", "url": "", "snippet": text})

    return results
