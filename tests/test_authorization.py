"""Cross-user authorization tests."""

import pytest
from httpx import ASGITransport, AsyncClient

from forge.main import create_app
from tests.conftest import make_second_user


@pytest.fixture
def app():
    return create_app()


@pytest.mark.asyncio
async def test_thread_cross_user_idor(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"password": "securepass1"},
        )
        token_a = reg.json()["access_token"]
        headers_a = {"Authorization": f"Bearer {token_a}"}

        thread = await client.post("/api/inference/threads", json={"title": "secret"}, headers=headers_a)
        tid = thread.json()["id"]

        _, token_b = await make_second_user()
        headers_b = {"Authorization": f"Bearer {token_b}"}

        res = await client.get(f"/api/inference/threads/{tid}/messages", headers=headers_b)
        assert res.status_code == 404

        res_owner = await client.get(f"/api/inference/threads/{tid}/messages", headers=headers_a)
        assert res_owner.status_code == 200


@pytest.mark.asyncio
async def test_export_job_cross_user(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"password": "securepass1"},
        )

        from forge.api.deps import get_db

        db = get_db()
        user = await db.get_sole_user()
        job = await db.create_export_job(user["id"], {"checkpoint": "x"})
        job_id = job["id"]

        _, token_b = await make_second_user("c@local.dev")
        headers_b = {"Authorization": f"Bearer {token_b}"}

        res = await client.get(f"/api/export/jobs/{job_id}/stream", headers=headers_b)
        assert res.status_code == 404
