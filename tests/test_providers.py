import pytest
from httpx import ASGITransport, AsyncClient

from forge.main import create_app
from tests.conftest import RETURN_TOKEN_HEADERS


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def authed_client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
        )
        token = reg.json()["access_token"]
        client.headers["Authorization"] = f"Bearer {token}"
        yield client


@pytest.mark.asyncio
async def test_provider_crud(authed_client):
    create = await authed_client.post(
        "/api/providers",
        json={
            "name": "Local vLLM",
            "provider_type": "vllm",
            "config": {"base_url": "http://127.0.0.1:8000"},
        },
    )
    assert create.status_code == 201
    pid = create.json()["id"]

    listing = await authed_client.get("/api/providers")
    assert len(listing.json()) == 1

    delete = await authed_client.delete(f"/api/providers/{pid}")
    assert delete.status_code == 200


@pytest.mark.asyncio
async def test_frontier_providers_rejected(authed_client):
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


@pytest.mark.asyncio
async def test_list_providers_excludes_cloud_gpu_credentials(authed_client):
    """Training cloud_gpu credential rows must not appear as chat providers."""
    from forge.services.training_service import cloud_gpu_provider_type

    listing = await authed_client.get("/api/providers")
    assert listing.status_code == 200
    allowed = {"local_chat", "remote_chat", "vllm", "vllm_cloud"}
    for row in listing.json():
        assert row["provider_type"] != cloud_gpu_provider_type()
        assert row["provider_type"] in allowed
