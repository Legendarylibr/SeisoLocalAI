"""MCP server access control."""

from __future__ import annotations

from fastapi import HTTPException

from forge.db.store import Database
from forge.mcp.client import McpManager


async def validate_mcp_server_ids(
    db: Database,
    mcp: McpManager,
    user_id: str,
    server_ids: list[str],
) -> list[str]:
    """Ensure each server is owned by user and connected in their pool."""
    if not server_ids:
        return []
    validated: list[str] = []
    for sid in server_ids:
        row = await db.get_mcp_server(sid, user_id)
        if not row:
            raise HTTPException(403, f"MCP server not found or not owned: {sid}")
        if not mcp.get(user_id, sid):
            raise HTTPException(400, f"MCP server not connected: {sid}")
        validated.append(sid)
    return validated
