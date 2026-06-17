"""MCP server management and connection routes."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from forge.api.deps import get_db, get_inference_orchestrator
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.mcp.client import McpServer
from forge.orchestrators.inference import InferenceOrchestrator
from forge.security.audit import audit_event
from forge.security.auth import get_current_user_id
from forge.security.mcp_env import sanitize_mcp_env
from seiso.security import sanitize_filename

_BLOCKED_ARG_PREFIXES = ("/etc", "/tmp", "/var", "/home", "/usr", "/System", "/private")
_INLINE_EXEC_FLAGS = frozenset({"-c", "-e", "-m", "--eval", "--exec"})
_INLINE_EXEC_CMDS = frozenset({"python", "python3", "node", "deno"})


def _validate_mcp_args(args: list[str], command: str) -> None:
    for arg in args:
        if ".." in arg:
            raise HTTPException(400, "Invalid args")
        if arg.startswith("/") and any(arg.startswith(p) for p in _BLOCKED_ARG_PREFIXES):
            raise HTTPException(400, "MCP args cannot reference system paths")
        if arg in ("-y", "--yes"):
            raise HTTPException(400, "Unpinned npx installs blocked — pin package versions instead of using -y")
    if command in _INLINE_EXEC_CMDS:
        for arg in args:
            if arg in _INLINE_EXEC_FLAGS or arg.startswith("-c") or arg.startswith("-e"):
                raise HTTPException(400, "Inline code execution in MCP args is not allowed")


def _require_mcp_enabled(settings: ForgeSettings) -> None:
    if not settings.allow_tools:
        raise HTTPException(403, "MCP is disabled — set SEISO_ALLOW_TOOLS=true to enable")


router = APIRouter(prefix="/mcp", tags=["mcp"])

_ALLOWED_COMMANDS = frozenset({"npx", "node", "python", "python3", "uvx", "deno"})


class McpServerCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    command: str = Field(min_length=1, description="Executable to spawn")
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


@router.get("/servers")
async def list_servers(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> list[dict]:
    rows = await db.list_mcp_servers(user_id)
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "command": r["command"],
            "args": json.loads(r["args_json"]),
            "enabled": bool(r["enabled"]),
            "created_at": r["created_at"],
        }
        for r in rows
    ]


@router.post("/servers", status_code=201)
async def create_server(
    body: McpServerCreate,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    _require_mcp_enabled(settings)
    cmd = body.command.strip().split("/")[-1]
    if cmd not in _ALLOWED_COMMANDS:
        raise HTTPException(400, f"Command must be one of: {sorted(_ALLOWED_COMMANDS)}")
    _validate_mcp_args(body.args, cmd)
    safe_env = sanitize_mcp_env(body.env)
    row = await db.create_mcp_server(user_id, sanitize_filename(body.name), cmd, body.args, safe_env)
    audit_event("mcp_server_create", user_id=user_id, command=cmd, name=body.name)
    return row


@router.post("/servers/{server_id}/connect")
async def connect_server(
    server_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[InferenceOrchestrator, Depends(get_inference_orchestrator)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    _require_mcp_enabled(settings)
    row = await db.get_mcp_server(server_id, user_id)
    if not row:
        raise HTTPException(404, "MCP server not found")

    cmd = row["command"]
    args = json.loads(row["args_json"])
    _validate_mcp_args(args, cmd)

    server = McpServer(
        id=row["id"],
        name=row["name"],
        command=cmd,
        args=args,
        env=json.loads(row["env_json"]),
        user_id=user_id,
    )
    await orchestrator.mcp.connect(user_id, server)
    tools = await server.list_tools()
    audit_event("mcp_connect", user_id=user_id, server_id=server_id, command=row["command"])
    return {"connected": True, "tools": [t.get("name") for t in tools]}


@router.post("/servers/{server_id}/disconnect")
async def disconnect_server(
    server_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[InferenceOrchestrator, Depends(get_inference_orchestrator)],
) -> dict:
    row = await db.get_mcp_server(server_id, user_id)
    if not row:
        raise HTTPException(404, "MCP server not found")
    await orchestrator.mcp.disconnect(user_id, server_id)
    audit_event("mcp_disconnect", user_id=user_id, server_id=server_id)
    return {"disconnected": True}


@router.get("/tools")
async def list_all_tools(
    user_id: Annotated[str, Depends(get_current_user_id)],
    orchestrator: Annotated[InferenceOrchestrator, Depends(get_inference_orchestrator)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> list[dict]:
    _require_mcp_enabled(settings)
    return await orchestrator.mcp.all_tools(user_id)


@router.delete("/servers/{server_id}")
async def delete_server(
    server_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[InferenceOrchestrator, Depends(get_inference_orchestrator)],
) -> dict:
    await orchestrator.mcp.disconnect(user_id, server_id)
    ok = await db.delete_mcp_server(server_id, user_id)
    if not ok:
        raise HTTPException(404, "MCP server not found")
    audit_event("mcp_server_delete", user_id=user_id, server_id=server_id)
    return {"deleted": True}
