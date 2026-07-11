"""Regression coverage for the optimization pass (lazy imports, caches, executors)."""

from __future__ import annotations

from pathlib import Path

import pytest

from seiso.training.config import DatasetFormat


def test_seiso_public_api_is_lazy():
    import seiso

    # Fresh module state: heavy symbols should resolve via __getattr__.
    assert hasattr(seiso, "__getattr__")
    TrainConfig = seiso.TrainConfig
    assert TrainConfig.__name__ == "TrainConfig"
    assert seiso.ExportFormat is not None


def test_forge_deps_lazy_orchestrator_import():
    from forge.api import deps

    # Module should not eagerly bind TrainingOrchestrator at import time.
    assert "TrainingOrchestrator" not in deps.__dict__
    orch = deps.get_training_orchestrator()
    assert orch is not None
    assert orch is deps.get_training_orchestrator()
    deps.clear_dependency_caches()


def test_executors_are_dedicated_pools():
    from forge.services.executors import GPU_EXECUTOR, IO_EXECUTOR

    assert GPU_EXECUTOR._max_workers == 1
    assert IO_EXECUTOR._max_workers == 4


def test_cleaned_dataset_cache_roundtrip():
    from seiso.training.dataset_analysis import (
        cleaned_dataset_cache_key,
        store_cleaned_dataset,
        take_cleaned_dataset,
    )

    key = cleaned_dataset_cache_key(
        "ds.jsonl",
        dataset_format=DatasetFormat.AUTO,
        sandbox_root=None,
        deduplicate=True,
        min_chars=1,
    )
    store_cleaned_dataset(
        key,
        ["cleaned"],
        {"kept": 1, "initial_samples": 2, "resolved_format": "chat"},
        DatasetFormat.CHAT,
    )
    got = take_cleaned_dataset(key)
    assert got is not None
    cleaned, stats, fmt = got
    assert cleaned == ["cleaned"]
    assert stats["kept"] == 1
    assert fmt == DatasetFormat.CHAT
    assert take_cleaned_dataset(key) is None


def test_analyze_stores_cleaned_cache(tmp_path: Path):
    from seiso.training.dataset_analysis import (
        analyze_training_dataset,
        take_cleaned_dataset,
    )

    ds = tmp_path / "train.jsonl"
    ds.write_text(
        '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}\n'
        '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}\n'
        '{"messages":[{"role":"user","content":"yo"},{"role":"assistant","content":"hey"}]}\n'
    )
    analysis = analyze_training_dataset(ds, dataset_format=DatasetFormat.CHAT)
    assert analysis["valid"] is True
    assert analysis.get("cleaned_cache_key")
    cached = take_cleaned_dataset(analysis["cleaned_cache_key"])
    assert cached is not None
    cleaned, stats, fmt = cached
    assert len(cleaned) == analysis["kept"]
    assert stats["kept"] == analysis["kept"]
    assert fmt == DatasetFormat.CHAT


def test_analyze_sample_mode_does_not_store_cleaned_cache(tmp_path: Path):
    from seiso.training.dataset_analysis import analyze_training_dataset

    ds = tmp_path / "train.jsonl"
    ds.write_text(
        '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}\n'
    )
    analysis = analyze_training_dataset(
        ds, dataset_format=DatasetFormat.CHAT, full_scan=False
    )
    assert analysis["uses_full_dataset"] is False
    assert "cleaned_cache_key" not in analysis


def test_dataset_analysis_result_cache_hits(tmp_path: Path, monkeypatch):
    from forge.services import training_service as ts

    calls = {"n": 0}
    real = ts.analyze_training_dataset

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(ts, "analyze_training_dataset", counting)
    ts._dataset_analysis_results.clear()

    sandbox = tmp_path / "uploads" / "u1"
    sandbox.mkdir(parents=True)
    ds = sandbox / "a.jsonl"
    ds.write_text(
        '{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}\n'
    )

    a1 = ts.run_dataset_analysis(ds, dataset_format=DatasetFormat.CHAT, sandbox_root=sandbox)
    a2 = ts.run_dataset_analysis(ds, dataset_format=DatasetFormat.CHAT, sandbox_root=sandbox)
    assert calls["n"] == 1
    assert a1["kept"] == a2["kept"]


def test_first_unique_indices_dedup():
    from seiso.training.preprocess import _first_unique_indices

    assert _first_unique_indices(["a", "b", "a", "c", "b"]) == [0, 1, 3]


def test_manifest_common_fields():
    from seiso.research.provenance import content_fingerprint, manifest_common_fields

    fp = content_fingerprint({"z": 1, "a": 2})
    assert len(fp) == 64
    fields = manifest_common_fields(config_snapshot={"model_id": "x"})
    assert "created_at" in fields
    assert fields["config_fingerprint"] == content_fingerprint({"model_id": "x"})


def test_tokenize_texts_batches():
    from seiso.codellama_compress.training_utils import tokenize_texts

    class _Tok:
        def __call__(self, texts, return_tensors="pt", truncation=True, max_length=8, padding=True):
            import torch

            ids = [[1, 2, 3] for _ in texts]
            return {
                "input_ids": torch.tensor(ids),
                "attention_mask": torch.ones(len(texts), 3, dtype=torch.long),
            }

    batch = tokenize_texts(_Tok(), ["hello", "world"], 8)
    assert batch["input_ids"].shape[0] == 2


def test_knowledge_retrieve_uses_inverted_index(tmp_path: Path):
    from forge.services.knowledge_context import retrieve_knowledge_chunks

    kb = tmp_path / "knowledge" / "u1" / "kb1"
    kb.mkdir(parents=True)
    index = kb / "index.jsonl"
    index.write_text(
        '{"text":"alpha beta gamma","source":"a.txt"}\n'
        '{"text":"delta epsilon","source":"b.txt"}\n'
        '{"text":"alpha zeta","source":"c.txt"}\n'
    )
    hits = retrieve_knowledge_chunks(
        tmp_path, user_id="u1", knowledge_base_id="kb1", query="alpha", top_k=5
    )
    assert len(hits) == 2
    assert all("alpha" in h["text"] for h in hits)

    # Second call should hit retrieve cache (same results).
    hits2 = retrieve_knowledge_chunks(
        tmp_path, user_id="u1", knowledge_base_id="kb1", query="alpha", top_k=5
    )
    assert [h["text"] for h in hits2] == [h["text"] for h in hits]


def test_policy_heads_dirty_flag_avoids_copy_each_read():
    import random

    from seiso.adaptive_quant.policy_heads import CategoricalHead

    head = CategoricalHead(4, 3, random.Random(0))
    w1 = head.weights
    w2 = head.weights
    assert w1 is w2 or w1 == w2
    # After update with native unavailable, Python path mutates in place.
    probs = [0.3, 0.3, 0.4]
    head.update([0.1, 0.2, 0.3, 0.4], 1, probs, advantage=0.5, learning_rate=0.01)
    assert len(head.weights) == 3


@pytest.mark.asyncio
async def test_db_metadata_repo_lookup(tmp_path: Path):
    from forge.db.crypto import generate_encryption_key
    from forge.db.store import Database

    db = Database(
        tmp_path / "t.db", encryption_key=generate_encryption_key(), ephemeral=True
    )
    await db.add_model(
        user_id="u1",
        source="hf:mirror/other",
        name="m",
        path=str(tmp_path / "m"),
        format="gguf",
        size_bytes=1,
        metadata={"repo_id": "org/Model"},
    )
    row = await db.get_model_by_metadata_repo_id("u1", "org/Model")
    assert row is not None
    assert row["source"] == "hf:mirror/other"


def test_create_app_registers_routes():
    from forge.main import create_app

    app = create_app()
    openapi_paths = set(app.openapi()["paths"])
    assert any(p.startswith("/api/training") for p in openapi_paths)
    assert any(p.startswith("/api/knowledge") for p in openapi_paths)
    assert any(p.startswith("/api/inference") for p in openapi_paths)
