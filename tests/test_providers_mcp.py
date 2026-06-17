import pytest
from httpx import ASGITransport, AsyncClient

from forge.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def authed_client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"email": "admin@local.dev", "password": "securepass1"},
        )
        token = reg.json()["access_token"]
        client.headers["Authorization"] = f"Bearer {token}"
        yield client


@pytest.mark.asyncio
async def test_provider_crud(authed_client):
    create = await authed_client.post(
        "/api/providers",
        json={"name": "OpenAI", "provider_type": "openai", "config": {"api_key": "sk-test"}},
    )
    assert create.status_code == 201
    pid = create.json()["id"]
    assert create.json()["config"]["api_key"] == "***"

    listing = await authed_client.get("/api/providers")
    assert len(listing.json()) == 1

    delete = await authed_client.delete(f"/api/providers/{pid}")
    assert delete.status_code == 200


@pytest.mark.asyncio
async def test_mcp_server_crud(authed_client, enable_tools):
    create = await authed_client.post(
        "/api/mcp/servers",
        json={"name": "test", "command": "npx", "args": ["@scope/pkg@1.0.0"]},
    )
    assert create.status_code == 201
    sid = create.json()["id"]

    listing = await authed_client.get("/api/mcp/servers")
    assert len(listing.json()) == 1

    delete = await authed_client.delete(f"/api/mcp/servers/{sid}")
    assert delete.status_code == 200


@pytest.mark.asyncio
async def test_mcp_rejects_unknown_command(authed_client, enable_tools):
    res = await authed_client.post(
        "/api/mcp/servers",
        json={"name": "bad", "command": "/bin/bash", "args": []},
    )
    assert res.status_code == 400
