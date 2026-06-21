from __future__ import annotations

from pathlib import Path

from forge.services.knowledge_context import format_knowledge_context, retrieve_knowledge_chunks
from forge.services.ollama_export import build_ollama_create_commands


def test_build_ollama_create_commands(tmp_path: Path):
    quant_dir = tmp_path / "q4_k_m"
    quant_dir.mkdir()
    (quant_dir / "model-q4_k_m.gguf").write_text("gguf")
    (quant_dir / "Modelfile").write_text("FROM ./model-q4_k_m.gguf\n")

    commands = build_ollama_create_commands(
        {"gguf_q4_k_m": str(quant_dir / "model-q4_k_m.gguf")},
        model_name="my-model",
    )
    assert commands == [f"ollama create my-model -f {quant_dir / 'Modelfile'}"]


def test_retrieve_knowledge_chunks_scores_overlap(tmp_path: Path):
    kb_dir = tmp_path / "knowledge" / "user1" / "docs"
    kb_dir.mkdir(parents=True)
    index = kb_dir / "index.jsonl"
    index.write_text(
        '{"text": "Seiso supports local GGUF chat", "source": "guide.txt"}\n'
        '{"text": "unrelated content about cooking", "source": "other.txt"}\n'
    )

    chunks = retrieve_knowledge_chunks(
        tmp_path,
        user_id="user1",
        knowledge_base_id="docs",
        query="local GGUF",
        top_k=2,
    )
    assert len(chunks) == 1
    assert "GGUF" in chunks[0]["text"]


def test_format_knowledge_context_includes_sources():
    text = format_knowledge_context([{"text": "Alpha beta", "source": "a.txt"}])
    assert "a.txt" in text
    assert "Alpha beta" in text
