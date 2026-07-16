"""Optional multi-GPU: local_chat / remote_chat + Compat API (vendor-neutral)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from forge.main import create_app
from forge.security.url_policy import validate_provider_base_url
from seiso.security import SecurityError


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def authed_client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"password": "securepass1"},
        )
        token = reg.json()["access_token"]
        client.headers["Authorization"] = f"Bearer {token}"
        yield client


@pytest.mark.asyncio
async def test_local_chat_provider(authed_client):
    create = await authed_client.post(
        "/api/providers",
        json={
            "name": "Local multi-GPU",
            "provider_type": "local_chat",
            "config": {"base_url": "http://127.0.0.1:8000"},
        },
    )
    assert create.status_code == 201
    assert create.json()["provider_type"] == "local_chat"


@pytest.mark.asyncio
async def test_vllm_alias_normalizes_to_local_chat(authed_client):
    """Legacy type vllm still works and is stored as local_chat."""
    create = await authed_client.post(
        "/api/providers",
        json={
            "name": "Legacy vLLM",
            "provider_type": "vllm",
            "config": {"base_url": "http://127.0.0.1:8000"},
        },
    )
    assert create.status_code == 201
    assert create.json()["provider_type"] == "local_chat"


@pytest.mark.asyncio
async def test_remote_chat_rejected_when_disabled(authed_client, monkeypatch):
    monkeypatch.delenv("SEISO_ALLOW_CLOUD_MULTIGPU", raising=False)
    monkeypatch.delenv("SEISO_ALLOW_CLOUD_PROVIDERS", raising=False)
    res = await authed_client.post(
        "/api/providers",
        json={
            "name": "Cloud TP",
            "provider_type": "remote_chat",
            "config": {
                "base_url": "https://example.com/v1",
                "model": "kimi-k3",
                "api_key": "sk-test",
                "tensor_parallel_size": 8,
            },
        },
    )
    assert res.status_code == 400


@pytest.mark.asyncio
async def test_remote_chat_accepted_when_enabled(authed_client, monkeypatch):
    monkeypatch.setenv("SEISO_ALLOW_CLOUD_MULTIGPU", "true")

    def _fake_resolve(host: str) -> list[str]:
        return ["93.184.216.34"]

    with patch("forge.security.url_policy._resolve_host", side_effect=_fake_resolve):
        res = await authed_client.post(
            "/api/providers",
            json={
                "name": "RunPod multi-GPU",
                "provider_type": "remote_chat",
                "config": {
                    "base_url": "https://example.com/v1",
                    "model": "kimi-k3",
                    "api_key": "sk-test",
                    "tensor_parallel_size": 8,
                    "gpu_count": 8,
                    "hoster": "runpod",
                },
            },
        )
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["provider_type"] == "remote_chat"
    assert body["config"]["model"] == "kimi-k3"
    assert body["config"]["api_key"] == "***"
    assert body["config"]["tensor_parallel_size"] == 8
    assert body["config"]["deployment_kind"] == "multi_gpu_remote"


@pytest.mark.asyncio
async def test_vllm_cloud_alias_normalizes_to_remote_chat(authed_client, monkeypatch):
    monkeypatch.setenv("SEISO_ALLOW_CLOUD_MULTIGPU", "true")

    def _fake_resolve(host: str) -> list[str]:
        return ["93.184.216.34"]

    with patch("forge.security.url_policy._resolve_host", side_effect=_fake_resolve):
        res = await authed_client.post(
            "/api/providers",
            json={
                "name": "Legacy cloud",
                "provider_type": "vllm_cloud",
                "config": {
                    "base_url": "https://example.com/v1",
                    "model": "m",
                    "api_key": "sk",
                },
            },
        )
    assert res.status_code == 201
    assert res.json()["provider_type"] == "remote_chat"


def test_remote_chat_rejects_loopback():
    with pytest.raises(SecurityError):
        validate_provider_base_url(
            "https://127.0.0.1:8000/v1", provider_type="remote_chat"
        )


def test_local_chat_still_allows_loopback():
    url = validate_provider_base_url(
        "http://127.0.0.1:8000/v1", provider_type="local_chat"
    )
    assert "127.0.0.1" in url


@pytest.mark.asyncio
async def test_managed_vllm_start_disabled_by_default(authed_client, monkeypatch):
    monkeypatch.delenv("SEISO_MANAGED_VLLM_ENABLED", raising=False)
    monkeypatch.delenv("SEISO_ALLOW_MANAGED_VLLM", raising=False)
    res = await authed_client.post(
        "/api/providers/managed-vllm/start",
        json={"model": "tiny-model"},
    )
    assert res.status_code == 400


@pytest.mark.asyncio
async def test_managed_vllm_status_always_available(authed_client):
    res = await authed_client.get("/api/providers/managed-vllm/status")
    assert res.status_code == 200
    data = res.json()
    assert "running" in data
    assert data["running"] is False


@pytest.mark.asyncio
async def test_compat_lists_provider_models_and_routes(authed_client, monkeypatch):
    """External agents see provider ids on /v1/models and can complete via them."""
    monkeypatch.setenv("SEISO_ALLOW_CLOUD_MULTIGPU", "true")

    def _fake_resolve(host: str) -> list[str]:
        return ["93.184.216.34"]

    with patch("forge.security.url_policy._resolve_host", side_effect=_fake_resolve):
        create = await authed_client.post(
            "/api/providers",
            json={
                "name": "Agent multi-GPU",
                "provider_type": "remote_chat",
                "config": {
                    "base_url": "https://example.com/v1",
                    "model": "agent-model-x",
                    "api_key": "sk-test",
                },
            },
        )
    assert create.status_code == 201
    pid = create.json()["id"]
    provider_model = f"provider:{pid}"

    from forge.config import get_settings

    settings = get_settings()
    key = settings.inference_api_key

    models = await authed_client.get(
        "/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    assert models.status_code == 200
    ids = {m["id"] for m in models.json()["data"]}
    assert provider_model in ids
    assert "agent-model-x" in ids

    with patch(
        "forge.orchestrators.inference.chat_completion",
        new=AsyncMock(return_value="hello from multi-gpu"),
    ):
        chat = await authed_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": provider_model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
    assert chat.status_code == 200, chat.text
    assert chat.json()["choices"][0]["message"]["content"] == "hello from multi-gpu"


@pytest.mark.asyncio
async def test_compat_default_still_requires_local_model(authed_client):
    from forge.config import get_settings

    key = get_settings().inference_api_key
    res = await authed_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert res.status_code in {400, 404, 409, 500}


@pytest.mark.asyncio
async def test_frontier_providers_still_rejected(authed_client):
    for ptype in ("openai", "anthropic"):
        res = await authed_client.post(
            "/api/providers",
            json={
                "name": ptype,
                "provider_type": ptype,
                "config": {"api_key": "sk-test"},
            },
        )
        assert res.status_code == 400


def test_suggest_tensor_parallel():
    from seiso.inference.managed_vllm import suggest_tensor_parallel

    assert suggest_tensor_parallel(1) == 1
    assert suggest_tensor_parallel(2) == 2
    assert suggest_tensor_parallel(3) == 2
    assert suggest_tensor_parallel(8) == 8
