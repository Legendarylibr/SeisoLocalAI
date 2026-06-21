"""RAG / knowledge base orchestrator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from forge.orchestrators.base import Orchestrator
from forge.security.audit import audit_event
from forge.services.knowledge_context import retrieve_knowledge_chunks
from forge.services.knowledge_paths import assert_ingest_source
from forge.tools.sanitize import wrap_tool_result
from seiso.security import safe_join
from seiso.security.deps import sha256_file


class KnowledgeOrchestrator(Orchestrator):
    kind = "knowledge"

    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64
    MAX_INGEST_BYTES = 50 * 1024 * 1024
    _index_cache: dict[str, tuple[float, list[dict]]] = {}

    async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        action = payload.get("action", "ingest")

        if action == "ingest":
            return await self._ingest(job_id, payload)
        if action == "retrieve":
            return await self._retrieve(job_id, payload)
        raise ValueError(f"Unknown action: {action}")

    def _kb_dir(self, user_id: str, kb_id: str) -> Path:
        return safe_join(self.sandbox_root, "knowledge", user_id, kb_id)

    async def _ingest(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        user_id = payload["user_id"]
        kb_id = payload["knowledge_base_id"]
        source = assert_ingest_source(self.sandbox_root, user_id, payload["source_path"])
        size = source.stat().st_size
        if size > self.MAX_INGEST_BYTES:
            raise ValueError(
                f"Source file exceeds {self.MAX_INGEST_BYTES // (1024 * 1024)} MiB ingest limit"
            )
        kb_dir = self._kb_dir(user_id, kb_id)
        kb_dir.mkdir(parents=True, exist_ok=True)

        text = source.read_text(encoding="utf-8", errors="replace")
        source_hash = sha256_file(source)
        chunks = self._chunk(text)
        self._emit_log(job_id, f"Ingested {source.name}: {len(chunks)} chunks")
        audit_event(
            "kb_ingest", user_id=user_id, kb_id=kb_id, source=str(source.name), chunks=len(chunks)
        )

        index_path = kb_dir / "index.jsonl"
        with index_path.open("a") as f:
            for i, chunk in enumerate(chunks):
                record = {
                    "id": hashlib.sha256(f"{kb_id}:{i}:{chunk[:32]}".encode()).hexdigest()[:16],
                    "text": chunk,
                    "source": str(source.name),
                    "source_path": str(source),
                    "source_sha256": source_hash,
                    "chunk_index": i,
                }
                f.write(json.dumps(record) + "\n")

        self._index_cache.pop(str(index_path), None)
        return {"chunk_count": len(chunks), "index_path": str(index_path)}

    def _load_index_chunks(self, index_path: Path) -> list[dict]:
        cache_key = str(index_path)
        try:
            mtime = index_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        cached = self._index_cache.get(cache_key)
        if cached and cached[0] == mtime:
            return cached[1]

        chunks: list[dict] = []
        with index_path.open() as f:
            for line in f:
                chunks.append(json.loads(line))
        self._index_cache[cache_key] = (mtime, chunks)
        return chunks

    async def _retrieve(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        user_id = payload["user_id"]
        kb_id = payload["knowledge_base_id"]
        query = payload["query"]
        top_k = payload.get("top_k", 5)

        chunks = retrieve_knowledge_chunks(
            self.sandbox_root,
            user_id=user_id,
            knowledge_base_id=kb_id,
            query=query,
            top_k=top_k,
        )
        results = [{**c, "text": wrap_tool_result(f"kb:{kb_id}", c["text"])} for c in chunks]
        self._emit_log(job_id, f"Retrieved {len(results)} chunks for query")
        return {"results": results}

    def _chunk(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        i = 0
        step = max(self.CHUNK_SIZE - self.CHUNK_OVERLAP, 1)
        while i < len(words):
            chunk = " ".join(words[i : i + self.CHUNK_SIZE])
            if chunk.strip():
                chunks.append(chunk)
            i += step
        return chunks
