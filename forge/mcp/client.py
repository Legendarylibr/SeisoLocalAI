"""MCP (Model Context Protocol) client — stdio JSON-RPC subprocess."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from forge.security.mcp_env import mcp_subprocess_env

logger = logging.getLogger(__name__)


@dataclass
class McpServer:
    id: str
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    user_id: str = ""
    _process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    _pending: dict[str, asyncio.Future] = field(default_factory=dict, repr=False)
    _reader_task: asyncio.Task | None = field(default=None, repr=False)

    async def start(self) -> None:
        if self._process and self._process.returncode is None:
            return
        self._process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=mcp_subprocess_env(self.env),
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        await self._request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "seiso-forge", "version": "0.1.0"}})
        await self._notify("notifications/initialized", {})

    async def stop(self) -> None:
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
        if self._reader_task:
            self._reader_task.cancel()

    async def list_tools(self) -> list[dict]:
        resp = await self._request("tools/list", {})
        return resp.get("tools", [])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        resp = await self._request("tools/call", {"name": name, "arguments": arguments})
        content = resp.get("content", [])
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        return "\n".join(texts) or json.dumps(resp)

    async def _request(self, method: str, params: dict) -> dict:
        if not self._process or not self._process.stdin:
            raise RuntimeError("MCP server not started")
        req_id = str(uuid.uuid4())
        msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        self._process.stdin.write((json.dumps(msg) + "\n").encode())
        await self._process.stdin.drain()
        try:
            return await asyncio.wait_for(fut, timeout=60)
        finally:
            self._pending.pop(req_id, None)

    async def _notify(self, method: str, params: dict) -> None:
        if not self._process or not self._process.stdin:
            return
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        self._process.stdin.write((json.dumps(msg) + "\n").encode())
        await self._process.stdin.drain()

    async def _read_loop(self) -> None:
        assert self._process and self._process.stdout
        while True:
            line = await self._process.stdout.readline()
            if not line:
                break
            try:
                data = json.loads(line.decode())
            except json.JSONDecodeError:
                continue
            req_id = data.get("id")
            if req_id and req_id in self._pending:
                fut = self._pending[req_id]
                if "error" in data:
                    fut.set_exception(RuntimeError(str(data["error"])))
                else:
                    fut.set_result(data.get("result", {}))


MAX_MCP_SERVERS_PER_USER = 8


class McpManager:
    """Per-user pool of connected MCP servers."""

    def __init__(self) -> None:
        self._servers: dict[str, dict[str, McpServer]] = {}

    def _user_pool(self, user_id: str) -> dict[str, McpServer]:
        return self._servers.setdefault(user_id, {})

    async def connect(self, user_id: str, server: McpServer) -> McpServer:
        pool = self._user_pool(user_id)
        if len(pool) >= MAX_MCP_SERVERS_PER_USER and server.id not in pool:
            raise RuntimeError(f"MCP server limit ({MAX_MCP_SERVERS_PER_USER}) reached for user")
        server.user_id = user_id
        await server.start()
        self._user_pool(user_id)[server.id] = server
        return server

    async def disconnect(self, user_id: str, server_id: str) -> None:
        pool = self._user_pool(user_id)
        srv = pool.pop(server_id, None)
        if srv:
            await srv.stop()

    def get(self, user_id: str, server_id: str) -> McpServer | None:
        return self._user_pool(user_id).get(server_id)

    async def all_tools(self, user_id: str) -> list[dict]:
        tools: list[dict] = []
        for sid, srv in self._user_pool(user_id).items():
            try:
                for t in await srv.list_tools():
                    tools.append({**t, "mcp_server_id": sid})
            except Exception as exc:
                logger.warning("MCP tools/list failed for %s: %s", sid, exc)
        return tools

    async def call(self, user_id: str, server_id: str, name: str, arguments: dict) -> str:
        srv = self.get(user_id, server_id)
        if not srv:
            raise KeyError(f"MCP server not connected: {server_id}")
        return await srv.call_tool(name, arguments)
