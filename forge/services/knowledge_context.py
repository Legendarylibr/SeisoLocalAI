"""Knowledge-base retrieval helpers for chat context injection."""

from __future__ import annotations

import json
from pathlib import Path

from seiso.security import safe_join


def retrieve_knowledge_chunks(
    data_dir: Path,
    *,
    user_id: str,
    knowledge_base_id: str,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Return top keyword-scored chunks from a user's knowledge base index."""
    kb_dir = safe_join(data_dir, "knowledge", user_id, knowledge_base_id)
    index_path = kb_dir / "index.jsonl"
    if not index_path.is_file():
        return []

    chunks: list[dict] = []
    with index_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    q_tokens = set(query.lower().split())
    if not q_tokens:
        return []

    scored: list[tuple[float, dict]] = []
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        t_tokens = set(text.lower().split())
        score = len(q_tokens & t_tokens) / max(len(q_tokens), 1)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def format_knowledge_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as a system-side context block."""
    if not chunks:
        return ""

    parts = [
        "Use the following reference excerpts from the user's knowledge base when answering. "
        "Prefer facts from these excerpts; say when the excerpts do not cover the question.",
        "",
    ]
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.get("source") or chunk.get("source_path") or "document"
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        parts.append(f"[{index}] ({source})\n{text}")
        parts.append("")

    return "\n".join(parts).strip()
