"""Tests for Hugging Face Hub model search."""

from __future__ import annotations

import pytest

from seiso.models.catalog import (
    HubSearchError,
    ModelFamily,
    diversify_by_family,
    get_by_repo,
    get_families,
    search_catalog,
)


def _sample_hub_rows() -> list[dict]:
    return [
        {
            "id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "downloads": 500_000,
            "createdAt": "2026-03-01T00:00:00.000Z",
            "pipeline_tag": "text-generation",
            "tags": [
                "gguf",
                "qwen3.6",
                "moe",
                "text-generation",
                "base_model:Qwen/Qwen3.6-35B-A3B",
            ],
        },
        {
            "id": "unsloth/Qwen3.6-27B-GGUF",
            "downloads": 450_000,
            "createdAt": "2026-03-01T00:00:00.000Z",
            "pipeline_tag": "text-generation",
            "tags": [
                "gguf",
                "qwen3.6",
                "text-generation",
                "base_model:Qwen/Qwen3.6-27B",
            ],
        },
        {
            "id": "unsloth/Qwen3.6-4B-GGUF",
            "downloads": 400_000,
            "createdAt": "2026-03-01T00:00:00.000Z",
            "pipeline_tag": "text-generation",
            "tags": [
                "gguf",
                "qwen3.6",
                "text-generation",
                "base_model:Qwen/Qwen3.6-4B",
            ],
        },
        {
            "id": "unsloth/Qwen3.5-4B-GGUF",
            "downloads": 390_000,
            "createdAt": "2025-11-01T00:00:00.000Z",
            "pipeline_tag": "text-generation",
            "tags": [
                "gguf",
                "qwen3.5",
                "text-generation",
                "base_model:Qwen/Qwen3.5-4B",
            ],
        },
        {
            "id": "bartowski/gemma-3-27b-it-GGUF",
            "downloads": 380_000,
            "createdAt": "2025-10-01T00:00:00.000Z",
            "pipeline_tag": "text-generation",
            "tags": ["gguf", "gemma", "vision", "base_model:google/gemma-3-27b-it"],
        },
        {
            "id": "bartowski/Devstral-Small-2507-GGUF",
            "downloads": 370_000,
            "createdAt": "2026-01-01T00:00:00.000Z",
            "pipeline_tag": "text-generation",
            "tags": [
                "gguf",
                "mistral",
                "code",
                "base_model:mistralai/Devstral-Small-2507",
            ],
        },
        {
            "id": "unsloth/DeepSeek-R1-0528-GGUF",
            "downloads": 360_000,
            "createdAt": "2026-02-01T00:00:00.000Z",
            "pipeline_tag": "text-generation",
            "tags": [
                "gguf",
                "deepseek",
                "moe",
                "base_model:deepseek-ai/DeepSeek-R1-0528",
            ],
        },
        {
            "id": "unsloth/gpt-oss-20b-GGUF",
            "downloads": 350_000,
            "createdAt": "2026-02-01T00:00:00.000Z",
            "pipeline_tag": "text-generation",
            "tags": ["gguf", "openai", "base_model:openai/gpt-oss-20b"],
        },
        {
            "id": "unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF",
            "downloads": 340_000,
            "createdAt": "2026-01-15T00:00:00.000Z",
            "pipeline_tag": "text-generation",
            "tags": [
                "gguf",
                "llama",
                "moe",
                "base_model:meta-llama/Llama-4-Scout-17B-16E-Instruct",
            ],
        },
        {
            "id": "unsloth/Kimi-K2-Instruct-GGUF",
            "downloads": 330_000,
            "createdAt": "2026-01-01T00:00:00.000Z",
            "pipeline_tag": "text-generation",
            "tags": ["gguf", "kimi", "moe", "base_model:moonshotai/Kimi-K2-Instruct"],
        },
        {
            "id": "AesSedai/Kimi-K2.7-Code-GGUF",
            "downloads": 320_000,
            "createdAt": "2026-03-10T00:00:00.000Z",
            "pipeline_tag": "text-generation",
            "tags": ["gguf", "kimi", "code", "base_model:moonshotai/Kimi-K2.7-Code"],
        },
        {
            "id": "unsloth/GLM-4.5-Air-GGUF",
            "downloads": 310_000,
            "createdAt": "2026-01-01T00:00:00.000Z",
            "pipeline_tag": "text-generation",
            "tags": ["gguf", "glm", "moe", "base_model:zai-org/GLM-4.5-Air"],
        },
        {
            "id": "bartowski/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
            "downloads": 300_000,
            "createdAt": "2026-01-01T00:00:00.000Z",
            "pipeline_tag": "text-generation",
            "tags": [
                "gguf",
                "mistral",
                "vision",
                "base_model:mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            ],
        },
        {
            "id": "unsloth/Qwen3-Coder-Next-GGUF",
            "downloads": 290_000,
            "createdAt": "2026-02-15T00:00:00.000Z",
            "pipeline_tag": "text-generation",
            "tags": ["gguf", "qwen", "code", "base_model:Qwen/Qwen3-Coder-Next"],
        },
        {
            "id": "unsloth/Qwen3-VL-2B-Instruct-GGUF",
            "downloads": 280_000,
            "createdAt": "2026-01-01T00:00:00.000Z",
            "pipeline_tag": "image-text-to-text",
            "tags": [
                "gguf",
                "qwen",
                "vision",
                "multimodal",
                "base_model:Qwen/Qwen3-VL-2B-Instruct",
            ],
        },
        {
            "id": "random-user/Qwen3.6-4B-GGUF",
            "downloads": 999_999,
            "createdAt": "2026-03-01T00:00:00.000Z",
            "pipeline_tag": "text-generation",
            "tags": [
                "gguf",
                "qwen3.6",
                "text-generation",
                "base_model:Qwen/Qwen3.6-4B",
            ],
        },
    ]


@pytest.fixture(autouse=True)
def _mock_hub_search(request, monkeypatch):
    if request.node.name in {
        "test_fetch_hub_page_returns_cursor",
        "test_hub_search_raises_on_rate_limit",
        "test_query_hub_page_omits_pipeline_tag_when_searching",
        "test_search_catalog_keeps_non_text_generation_when_querying",
        "test_search_catalog_browse_keeps_hub_rows_without_task_filter",
        "test_search_catalog_browse_still_filters_task",
    }:
        yield
        return

    def _query_page(**kwargs):
        rows = _sample_hub_rows()
        search = (kwargs.get("query") or "").lower().strip()
        if search:
            tokens = search.split()

            def _matches(row: dict) -> bool:
                hay = f"{row['id']} {' '.join(row.get('tags') or [])}".lower()
                return all(tok in hay for tok in tokens)

            rows = [row for row in rows if _matches(row)]
        limit = kwargs.get("limit", len(rows))
        return rows[:limit], None

    monkeypatch.setattr("seiso.models.catalog._query_hub_page", _query_page)
    yield


def test_hub_search_includes_popular_gguf_repos():
    result = search_catalog()
    repos = {m["repo_id"] for m in result.models}
    assert "random-user/Qwen3.6-4B-GGUF" in repos
    assert "AesSedai/Kimi-K2.7-Code-GGUF" in repos


def test_hub_search_returns_gguf_models():
    result = search_catalog()
    assert len(result.models) >= 10
    repos = {m["repo_id"] for m in result.models}
    assert "unsloth/Qwen3.6-35B-A3B-GGUF" in repos
    assert all(m.get("gguf_repo") for m in result.models)


def test_hub_search_keeps_all_variants():
    result = search_catalog()
    repos = {m["repo_id"] for m in result.models}
    assert "unsloth/Qwen3.6-4B-GGUF" in repos
    assert "unsloth/Qwen3.5-4B-GGUF" in repos


def test_get_families_is_static():
    families = set(get_families())
    for brand in ("llama", "qwen", "gemma", "mistral", "deepseek", "kimi", "glm"):
        assert brand in families


def test_hub_search_query():
    results = search_catalog("qwen coder").models
    assert any("coder" in m["repo_id"].lower() for m in results)


def test_hub_search_filter_family():
    results = search_catalog(family="gemma").models
    assert all(m["family"] == "gemma" for m in results)


def test_get_by_repo_fetches_from_hub(monkeypatch):
    class FakeInfo:
        id = "unsloth/Qwen3.6-4B-GGUF"
        downloads = 400_000
        pipeline_tag = "text-generation"
        tags = ["gguf", "qwen3.6", "text-generation"]
        created_at = None

    class FakeApi:
        def __init__(self, token=None):
            _ = token

        def model_info(self, repo_id):
            _ = repo_id
            return FakeInfo()

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    entry = get_by_repo("unsloth/Qwen3.6-4B-GGUF")
    assert entry is not None
    assert entry.family == ModelFamily.QWEN


def test_get_by_repo_returns_none_on_transport_error(monkeypatch):
    class FakeApi:
        def __init__(self, token=None):
            _ = token

        def model_info(self, repo_id):
            _ = repo_id
            import httpx

            raise httpx.ProxyError("403 Forbidden")

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    assert get_by_repo("org/Model") is None


def test_get_by_repo_returns_none_on_requests_connection_error(monkeypatch):
    class FakeApi:
        def __init__(self, token=None):
            _ = token

        def model_info(self, repo_id):
            _ = repo_id
            import requests

            raise requests.exceptions.ConnectionError("connection refused")

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    assert get_by_repo("org/Model") is None


def test_hub_search_priority_order():
    results = search_catalog().models
    assert results[0]["downloads"] >= results[-1]["downloads"]


def test_hub_search_ranks_exact_match():
    results = search_catalog("qwen 3.6").models
    assert "qwen" in results[0]["repo_id"].lower()


def test_hub_search_chat_includes_code_models():
    results = search_catalog(task="chat").models
    repos = {m["repo_id"] for m in results}
    assert "unsloth/Qwen3.6-27B-GGUF" in repos
    assert "unsloth/Qwen3-Coder-Next-GGUF" in repos


def test_diversify_by_family_interleaves_brands():
    models = search_catalog().models
    diversified = diversify_by_family(models[:24])
    first_families = [m["family"] for m in diversified[:12]]
    assert len(set(first_families)) >= 5


def test_hub_row_to_entry_accepts_any_text_generation_repo():
    from seiso.models.catalog import _hub_row_to_entry

    entry = _hub_row_to_entry(
        {
            "id": "Qwen/Qwen2.5-0.5B-Instruct",
            "downloads": 999,
            "tags": ["safetensors", "text-generation"],
            "pipeline_tag": "text-generation",
        }
    )
    assert entry is not None
    assert entry.repo_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert entry.gguf_repo is None

    gguf_entry = _hub_row_to_entry(
        {
            "id": "local-owner/Kimi-DFlash",
            "downloads": 999,
            "tags": ["gguf", "text-generation"],
            "pipeline_tag": "text-generation",
        }
    )
    assert gguf_entry is not None
    assert gguf_entry.gguf_repo == "local-owner/Kimi-DFlash"


def test_search_catalog_uses_hub_cursor(monkeypatch):
    def _query_page(**kwargs):
        rows = _sample_hub_rows()
        if kwargs.get("cursor") == "page2":
            return rows[2:4], None
        return rows[:2], "page2"

    monkeypatch.setattr("seiso.models.catalog._query_hub_page", _query_page)

    page_one = search_catalog(limit=2)
    assert len(page_one.models) == 2
    assert page_one.next_cursor == "page2"

    page_two = search_catalog(limit=2, cursor="page2")
    assert len(page_two.models) == 2
    assert page_two.next_cursor is None
    assert page_one.models[0]["repo_id"] != page_two.models[0]["repo_id"]


def test_hub_search_raises_on_rate_limit(monkeypatch):
    def _fail(**kwargs):
        _ = kwargs
        raise HubSearchError("rate limit", status_code=429)

    monkeypatch.setattr("seiso.models.catalog._query_hub_page", _fail)
    with pytest.raises(HubSearchError) as exc:
        search_catalog()
    assert exc.value.status_code == 429


def test_fetch_hub_page_returns_cursor(monkeypatch):
    class FakeResponse:
        headers = {
            "link": '<https://huggingface.co/api/models?filter=gguf&cursor=abc123>; rel="next"',
        }

        @staticmethod
        def json():
            return [{"id": "org/model-a", "downloads": 10, "tags": ["gguf"]}]

    class FakeSession:
        def get(self, path, params=None, headers=None):
            _ = (path, params, headers)
            return FakeResponse()

    monkeypatch.setattr("huggingface_hub.utils.get_session", lambda: FakeSession())
    monkeypatch.setattr("huggingface_hub.utils.hf_raise_for_status", lambda _r: None)
    monkeypatch.setattr("huggingface_hub.utils.build_hf_headers", lambda token=None: {})

    from seiso.models.catalog import _fetch_hub_page

    rows, next_cursor = _fetch_hub_page(filter_tag="gguf", limit=1)
    assert len(rows) == 1
    assert next_cursor == "abc123"


def test_query_hub_page_omits_pipeline_tag_when_searching(monkeypatch):
    captured: list[dict] = []

    def _fetch(**kwargs):
        captured.append(kwargs)
        return [], None

    monkeypatch.setattr("seiso.models.catalog._fetch_hub_page", _fetch)
    from seiso.models.catalog import _query_hub_page

    _query_hub_page(query="meta-llama/Llama-3.1-8B", limit=20)
    assert captured[0].get("pipeline_tag") is None
    assert "Llama-3.1-8B" in (captured[0].get("search") or "")

    captured.clear()
    _query_hub_page(query="", limit=50)
    # Empty browse uses text-generation so Hub/Chat show LLM model names.
    assert captured[0].get("pipeline_tag") == "text-generation"


def test_search_catalog_keeps_non_text_generation_when_querying(monkeypatch):
    def _query_page(**kwargs):
        return [
            {
                "id": "org/whisper-large",
                "downloads": 10_000,
                "pipeline_tag": "automatic-speech-recognition",
                "tags": ["asr"],
            },
            {
                "id": "org/chat-model",
                "downloads": 50_000,
                "pipeline_tag": "text-generation",
                "tags": ["text-generation"],
            },
        ], None

    monkeypatch.setattr("seiso.models.catalog._query_hub_page", _query_page)
    results = search_catalog("whisper").models
    repos = {m["repo_id"] for m in results}
    assert "org/whisper-large" in repos
    assert "org/chat-model" in repos


def test_search_catalog_browse_keeps_hub_rows_without_task_filter(monkeypatch):
    def _query_page(**kwargs):
        return [
            {
                "id": "org/whisper-large",
                "downloads": 90_000,
                "pipeline_tag": "automatic-speech-recognition",
                "tags": ["asr"],
            },
            {
                "id": "org/chat-model",
                "downloads": 50_000,
                "pipeline_tag": "text-generation",
                "tags": ["text-generation"],
            },
        ], None

    monkeypatch.setattr("seiso.models.catalog._query_hub_page", _query_page)
    # Without a task filter, mapped Hub rows are kept (pipeline is applied upstream).
    results = search_catalog().models
    repos = {m["repo_id"] for m in results}
    assert "org/whisper-large" in repos
    assert "org/chat-model" in repos
    assert results[0]["repo_id"] == "org/whisper-large"


def test_search_catalog_browse_still_filters_task(monkeypatch):
    def _query_page(**kwargs):
        return [
            {
                "id": "org/vision-model",
                "downloads": 10_000,
                "pipeline_tag": "image-text-to-text",
                "tags": ["vision"],
            },
            {
                "id": "org/chat-model",
                "downloads": 50_000,
                "pipeline_tag": "text-generation",
                "tags": ["text-generation"],
            },
        ], None

    monkeypatch.setattr("seiso.models.catalog._query_hub_page", _query_page)
    results = search_catalog(task="chat").models
    repos = {m["repo_id"] for m in results}
    assert "org/chat-model" in repos
    assert "org/vision-model" not in repos
