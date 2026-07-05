"""Seiso Code workspace API — files, search, git, terminal, local agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from forge.api.deps import get_db, get_inference_orchestrator
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.inference import InferenceOrchestrator
from forge.security.auth import get_current_user_id
from forge.security.code_policy import security_snapshot
from forge.services.web_search import web_search_available, web_search_provider
from forge.services import code_workspace as ws
from forge.services.code_agent import stream_code_agent_events
from forge.services.code_agent_messages import build_trusted_code_agent_messages
from forge.services.hardware import hardware_profile, hardware_summary
from seiso.security import SecurityError

router = APIRouter(prefix="/code", tags=["code"])


class SaveFileRequest(BaseModel):
    path: str = Field(min_length=1, max_length=4096)
    content: str = ""
    destructive_acknowledged: bool = False


class CreateEntryRequest(BaseModel):
    path: str = Field(min_length=1, max_length=4096)
    kind: Literal["file", "dir"] = "file"
    content: str = ""
    destructive_acknowledged: bool = False


class TerminalRequest(BaseModel):
    command: str = Field(min_length=1, max_length=8192)
    cwd: str | None = None
    destructive_acknowledged: bool = False


class AgentMessage(BaseModel):
    role: Literal["user"]
    content: str = Field(min_length=1, max_length=65536)


class CodeAgentRequest(BaseModel):
    session_id: str | None = None
    model_id: str | None = None
    messages: list[AgentMessage] = Field(default_factory=list)
    enabled_tools: list[str] = Field(default_factory=list)
    context_path: str | None = None
    dangerous_tools_acknowledged: bool = False
    network_tools_acknowledged: bool = False
    temperature: float = Field(default=0.2, ge=0, le=1)
    max_tokens: int = Field(default=2048, ge=64, le=8192)


def _root(settings: ForgeSettings) -> Path:
    try:
        return ws.resolve_code_workspace(settings)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


def _handle(exc: Exception) -> HTTPException:
    if isinstance(exc, SecurityError):
        return HTTPException(403, str(exc))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(404, str(exc))
    if isinstance(exc, FileExistsError):
        return HTTPException(409, str(exc))
    if isinstance(exc, PermissionError):
        return HTTPException(403, str(exc))
    if isinstance(exc, NotADirectoryError):
        return HTTPException(400, "Working directory is not a folder")
    if isinstance(exc, ValueError):
        return HTTPException(400, str(exc))
    return HTTPException(500, str(exc))


@router.get("/security")
async def get_code_security(
    settings: Annotated[ForgeSettings, Depends(get_settings)],
    _user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict:
    snapshot = security_snapshot()
    snapshot["web_search_enabled"] = web_search_available(settings)
    snapshot["web_search_provider"] = web_search_provider(settings) if settings.web_search_enabled else None
    return snapshot


@router.get("/workspace")
async def get_workspace(
    _user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    root = _root(settings)
    return ws.workspace_snapshot(root, settings)


@router.get("/tree")
async def get_tree(
    _user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
    path: str = Query(default=""),
) -> dict:
    root = _root(settings)
    try:
        entries = ws.list_tree(root, path)
    except Exception as exc:
        raise _handle(exc) from exc
    return {"path": path, "entries": entries}


@router.get("/file")
async def get_file(
    _user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
    path: str = Query(min_length=1),
) -> dict:
    root = _root(settings)
    try:
        return ws.read_file(root, settings, path)
    except Exception as exc:
        raise _handle(exc) from exc


@router.put("/file")
async def save_file(
    body: SaveFileRequest,
    _user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    root = _root(settings)
    try:
        return ws.write_file(
            root, settings, body.path, body.content, destructive_ack=body.destructive_acknowledged
        )
    except Exception as exc:
        raise _handle(exc) from exc


@router.post("/file")
async def create_file(
    body: CreateEntryRequest,
    _user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    root = _root(settings)
    try:
        return ws.create_entry(
            root,
            settings,
            body.path,
            body.kind,
            body.content,
            destructive_ack=body.destructive_acknowledged,
        )
    except Exception as exc:
        raise _handle(exc) from exc


@router.delete("/file")
async def delete_file(
    _user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
    path: str = Query(min_length=1),
    destructive_acknowledged: bool = Query(default=False),
) -> dict:
    root = _root(settings)
    try:
        return ws.delete_entry(root, path, destructive_ack=destructive_acknowledged)
    except Exception as exc:
        raise _handle(exc) from exc


@router.get("/search")
async def search_files(
    _user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
    q: str = Query(min_length=1),
    limit: int = Query(default=40, ge=1, le=200),
) -> dict:
    root = _root(settings)
    return ws.search_code(root, q, limit)


@router.get("/diff")
async def get_diff(
    _user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
    path: str | None = Query(default=None),
) -> dict:
    root = _root(settings)
    return ws.file_diff(root, path)


@router.post("/terminal")
async def run_terminal(
    body: TerminalRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    root = _root(settings)
    try:
        return ws.run_terminal(
            root, body.command, body.cwd, user_id=user_id, destructive_ack=body.destructive_acknowledged
        )
    except Exception as exc:
        raise _handle(exc) from exc


@router.get("/agent/status")
async def code_agent_status(
    _user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict:
    from forge.services.code_agent import max_concurrent_code_agents, waiting_agent_count

    profile = hardware_profile()
    return {
        "waiting": waiting_agent_count(),
        "max_concurrent": max_concurrent_code_agents(profile),
        "hardware_summary": hardware_summary(profile),
    }


@router.post("/agent")
async def run_code_agent_route(
    body: CodeAgentRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
    orchestrator: Annotated[InferenceOrchestrator, Depends(get_inference_orchestrator)],
):
    if not settings.allow_tools:
        raise HTTPException(403, "Local agents require tools to be enabled on this Forge server")

    if not body.messages:
        raise HTTPException(400, "messages are required")
    if not body.enabled_tools:
        raise HTTPException(400, "Enable at least one workspace tool")

    root = _root(settings)
    messages, _user_content, session_id = build_trusted_code_agent_messages(
        settings,
        user_id=user_id,
        session_id=body.session_id,
        client_messages=[msg.model_dump() for msg in body.messages],
    )

    async def event_gen():
        async for event in stream_code_agent_events(
            orchestrator=orchestrator,
            db=db,
            user_id=user_id,
            settings=settings,
            root=root,
            model_id=body.model_id,
            enabled_tools=body.enabled_tools,
            messages=messages,
            context_path=body.context_path,
            session_id=session_id,
            dangerous_tools_acknowledged=body.dangerous_tools_acknowledged,
            network_tools_acknowledged=body.network_tools_acknowledged,
        ):
            yield event

    return EventSourceResponse(event_gen())
