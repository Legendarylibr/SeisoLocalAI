"""Tests for GGUF repo resolution before Hub chat downloads."""

from __future__ import annotations

from forge.services import hf_hub


def test_gguf_mirror_candidates_include_bartowski():
    from seiso.models.trusted_gguf import gguf_mirror_candidates

    candidates = gguf_mirror_candidates("Qwen/Qwen3.6-27B")
    assert any("Qwen3.6-27B" in c for c in candidates)
    assert candidates[0].startswith("unsloth/")


def test_resolve_gguf_repo_uses_explicit_gguf_repo(monkeypatch):
    class Entry:
        gguf_repo = "bartowski/Custom-GGUF"
        quant = "Q4_K_M"

    resolved = hf_hub.resolve_gguf_repo("Qwen/Qwen3.6-27B", entry=Entry())
    assert resolved == "bartowski/Custom-GGUF"


def test_resolve_gguf_repo_falls_back_to_mirror(monkeypatch):
    monkeypatch.setattr(hf_hub, "repo_has_gguf", lambda repo_id, **_: repo_id.endswith("-GGUF"))
    resolved = hf_hub.resolve_gguf_repo("meta-llama/Llama-3.1-8B-Instruct")
    assert resolved.endswith("-GGUF")


def test_resolve_gguf_repo_ignores_dflash_draft_candidates(monkeypatch):
    monkeypatch.setattr(hf_hub, "repo_has_gguf", lambda repo_id, **_: repo_id == "bartowski/Kimi-GGUF")
    monkeypatch.setattr(
        hf_hub,
        "search_huggingface_gguf_repos",
        lambda **_k: [
            {"repo_id": "bartowski/Kimi-DFlash"},
            {"repo_id": "bartowski/Kimi-GGUF"},
        ],
    )
    hf_hub._GGUF_REPO_CACHE.clear()

    resolved = hf_hub.resolve_gguf_repo("org/Kimi")

    assert resolved == "bartowski/Kimi-GGUF"


def test_search_huggingface_datasets_parses_api_response(monkeypatch):
    payload = [
        {
            "id": "HuggingFaceH4/no_robots",
            "downloads": 12345,
            "tags": ["sft", "chat"],
        }
    ]

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            import json

            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(hf_hub.urllib.request, "urlopen", lambda *_a, **_k: FakeResponse())
    rows = hf_hub.search_huggingface_datasets(query="no_robots", limit=5)
    assert rows == [
        {
            "repo_id": "HuggingFaceH4/no_robots",
            "name": "no_robots",
            "downloads": 12345,
            "tags": ["sft", "chat"],
        }
    ]


def test_search_huggingface_datasets_empty_query():
    assert hf_hub.search_huggingface_datasets(query="  ") == []


def test_resolve_gguf_artifact_composes_mirror_file_and_size(monkeypatch):
    class Entry:
        gguf_repo = None
        quant = "Q4_K_M"

    monkeypatch.setattr(hf_hub, "resolve_gguf_repo", lambda *_a, **_k: "mirror/Model-GGUF")
    monkeypatch.setattr(
        hf_hub,
        "_list_repo_files",
        lambda *_a, **_k: ["Model-Q4_K_M.gguf", "Model-Q8_0.gguf", "mmproj-Q6_K.gguf"],
    )
    monkeypatch.setattr(hf_hub, "get_gguf_file_size_bytes", lambda *_a, **_k: 5_000_000_000)

    artifact = hf_hub.resolve_gguf_artifact("org/Qwen3.6-35B-A3B", entry=Entry(), use_cache=False)
    assert artifact["gguf_repo"] == "mirror/Model-GGUF"
    assert artifact["filename"] == "Model-Q4_K_M.gguf"
    assert artifact["size_bytes"] == 5_000_000_000


def test_pick_gguf_file_prefers_active_moe_quant():
    files = [
        "Qwen3.6-35B-Q4_K_M.gguf",
        "Qwen3.6-35B-A3B-Q4_K_M.gguf",
        "mmproj-Q6_K.gguf",
    ]
    picked = hf_hub._pick_gguf_file(files, preferred_quant="Q4_K_M", repo_id="Qwen/Qwen3.6-35B-A3B")
    assert picked == "Qwen3.6-35B-A3B-Q4_K_M.gguf"


def test_resolve_gguf_repo_prefers_trusted_search_over_untrusted(monkeypatch):
    monkeypatch.setattr(hf_hub, "repo_has_gguf", lambda repo_id, **_: repo_id == "bartowski/Kimi-GGUF")
    monkeypatch.setattr(
        hf_hub,
        "search_huggingface_gguf_repos",
        lambda **_k: [
            {"repo_id": "random-user/Kimi-GGUF", "downloads": 999_999},
            {"repo_id": "bartowski/Kimi-GGUF", "downloads": 10},
        ],
    )
    hf_hub._GGUF_REPO_CACHE.clear()

    resolved = hf_hub.resolve_gguf_repo("org/Kimi")

    assert resolved == "bartowski/Kimi-GGUF"


def test_resolve_gguf_repo_raises_when_missing(monkeypatch):
    monkeypatch.setattr(hf_hub, "repo_has_gguf", lambda *_a, **_k: False)
    monkeypatch.setattr(hf_hub, "search_huggingface_gguf_repos", lambda **_k: [])
    hf_hub._GGUF_REPO_CACHE.clear()
    try:
        hf_hub.resolve_gguf_repo("org/NoGgufModel")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "No trusted GGUF quant repo found" in str(exc)


def test_resolve_gguf_repo_uses_cache(monkeypatch):
    hf_hub._GGUF_REPO_CACHE.clear()
    calls = {"n": 0}

    def _has_gguf(repo_id, **_) -> bool:
        calls["n"] += 1
        return repo_id == "meta-llama/Llama-3.1-8B-Instruct"

    monkeypatch.setattr(hf_hub, "repo_has_gguf", _has_gguf)
    first = hf_hub.resolve_gguf_repo("meta-llama/Llama-3.1-8B-Instruct")
    second = hf_hub.resolve_gguf_repo("meta-llama/Llama-3.1-8B-Instruct")
    assert first == second == "meta-llama/Llama-3.1-8B-Instruct"
    assert calls["n"] == 1


def test_first_repo_with_gguf_preserves_candidate_order(monkeypatch):
    monkeypatch.setattr(hf_hub, "repo_has_gguf", lambda repo_id, **_: repo_id in {"second/repo", "third/repo"})

    resolved = hf_hub._first_repo_with_gguf(["first/repo", "second/repo", "third/repo"])

    assert resolved == "second/repo"


def test_download_gguf_skips_size_lookup_when_total_known(monkeypatch, tmp_path):
    size_calls = {"n": 0}

    def _size(*_a, **_k):
        size_calls["n"] += 1
        return 123

    monkeypatch.setattr(hf_hub, "get_gguf_file_size_bytes", _size)

    cached = tmp_path / "model.gguf"
    cached.write_bytes(b"gguf")
    monkeypatch.setattr(hf_hub, "_with_download_retries", lambda _fn, **_k: str(cached.resolve()))

    progress_events: list[dict] = []
    hf_hub.download_gguf(
        "mirror/Model-GGUF",
        cache_dir=tmp_path,
        token=None,
        filename="Model-Q4_K_M.gguf",
        total_bytes=5_000_000_000,
        on_progress=progress_events.append,
    )
    assert size_calls["n"] == 0
    assert progress_events == []
