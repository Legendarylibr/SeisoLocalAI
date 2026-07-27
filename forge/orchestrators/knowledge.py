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
from forge.tools.sanitize import prepare_kb_chunk_text, wrap_kb_reference
from seiso.security import safe_join
from seiso.security.deps import sha256_file


class KnowledgeOrchestrator(Orchestrator):
    kind = "knowledge"

    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64
    MAX_INGEST_BYTES = 50 * 1024 * 1024

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
        safe_chunks: list[str] = []
        skipped = 0
        for chunk in chunks:
            body, flagged = prepare_kb_chunk_text(chunk)
            if flagged or not body:
                skipped += 1
                continue
            safe_chunks.append(body)
        chunks = safe_chunks
        if skipped:
            self._emit_log(
                job_id,
                f"Skipped {skipped} instruction-like or empty chunk(s) from {source.name}",
            )
        self._emit_log(job_id, f"Ingested {source.name}: {len(chunks)} chunks")
        audit_event(
            "kb_ingest",
            user_id=user_id,
            kb_id=kb_id,
            source=str(source.name),
            chunks=len(chunks),
        )

        index_path = kb_dir / "index.jsonl"
        # Replace prior chunks from the same source so re-ingest does not duplicate.
        kept: list[dict[str, Any]] = []
        if index_path.is_file():
            with index_path.open(encoding="utf-8") as existing:
                for line in existing:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (
                        rec.get("source_sha256") == source_hash
                        or rec.get("source_path") == str(source)
                        or rec.get("source") == str(source.name)
                    ):
                        continue
                    kept.append(rec)
        with index_path.open("w", encoding="utf-8") as f:
            for rec in kept:
                f.write(json.dumps(rec) + "\n")
            for i, chunk in enumerate(chunks):
                record = {
                    "id": hashlib.sha256(
                        f"{kb_id}:{source_hash}:{i}:{chunk[:32]}".encode()
                    ).hexdigest()[:16],
                    "text": chunk,
                    "source": str(source.name),
                    "source_path": str(source),
                    "source_sha256": source_hash,
                    "chunk_index": i,
                    "instruction_flagged": False,
                }
                f.write(json.dumps(record) + "\n")

        return {"chunk_count": len(chunks), "index_path": str(index_path)}

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
        results = [{**c, "text": wrap_kb_reference(f"kb:{kb_id}", c["text"])} for c in chunks]
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
