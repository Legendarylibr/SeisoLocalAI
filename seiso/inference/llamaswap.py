"""Isolated GGUF sidecar clients (Ollama-first, llama-swap fallback).

Runtime health/engine/status lives in ``sidecar_runtime``; Ollama model registration
lives in ``ollama_registry``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from seiso.env import env_str
from seiso.inference.sidecar_runtime import (
    DEFAULT_LLAMASWAP_URL,
    DEFAULT_OLLAMA_URL,
    LlamaSwapRuntime,
    SidecarRuntime,
    llamaswap_enabled,
    llamaswap_health_ok,
    llamaswap_setup_hint,
    llamaswap_status,
    llamaswap_url,
    ollama_cli_host,
    ollama_health_ok,
    ollama_url,
    preferred_llamaswap_engine,
    preferred_sidecar_engine,
    sidecar_enabled,
    sidecar_setup_hint,
    sidecar_stack_ready,
    sidecar_status,
)
from seiso.inference.streaming import StreamToken

# Re-export runtime surface (implemented in sidecar_runtime).
__all__ = [
    "DEFAULT_LLAMASWAP_URL",
    "DEFAULT_OLLAMA_URL",
    "IsolatedGgufClient",
    "LlamaSwapClient",
    "LlamaSwapRuntime",
    "OllamaClient",
    "SidecarRuntime",
    "create_isolated_gguf_client",
    "llamaswap_enabled",
    "llamaswap_health_ok",
    "llamaswap_model_name",
    "llamaswap_setup_hint",
    "llamaswap_status",
    "llamaswap_url",
    "ollama_cli_host",
    "ollama_health_ok",
    "ollama_url",
    "preferred_llamaswap_engine",
    "preferred_sidecar_engine",
    "sidecar_enabled",
    "sidecar_setup_hint",
    "sidecar_stack_ready",
    "sidecar_status",
]


class IsolatedGgufClient(Protocol):
    engine: str

    def ensure_ready(self) -> None: ...

    def complete(self, payload: dict[str, Any], model_path: str) -> str: ...

    def stream(
        self,
        payload: dict[str, Any],
        model_path: str,
        *,
        should_stop,
    ) -> Iterator[StreamToken]: ...

    def release_external_memory(
        self, model_path: str | None = None
    ) -> tuple[bool, str | None]: ...


def llamaswap_model_name(model_path: str) -> str:
    """Resolve the model identifier expected by the llama-swap OpenAI API."""
    override = env_str("SEISO_LLAMASWAP_MODEL", "").strip()
    if override:
        return override
    return str(Path(model_path).expanduser())


def create_isolated_gguf_client(
    *, url: str | None = None, engine: str | None = None
) -> IsolatedGgufClient:
    """Return OllamaClient when Ollama is the active engine, else LlamaSwapClient."""
    selected = engine or preferred_sidecar_engine()
    if selected == "ollama":
        if not ollama_health_ok():
            raise RuntimeError(
                f"Ollama is not reachable at {ollama_url()}. "
                f"{sidecar_setup_hint(engine='ollama')}"
            )
        return OllamaClient(url=url)
    return LlamaSwapClient(url=url, engine="llamacpp")


class OllamaClient:
    """OpenAI-compatible client for Ollama's local /v1/chat/completions API."""

    def __init__(self, *, url: str | None = None) -> None:
        self.url = (url or ollama_url()).rstrip("/")
        self.engine = "ollama"

    def ensure_ready(self) -> None:
        if not ollama_health_ok(url=self.url):
            raise RuntimeError(
                f"Ollama is not reachable at {self.url}. "
                f"{sidecar_setup_hint(engine='ollama')}"
            )

    def complete(self, payload: dict[str, Any], model_path: str) -> str:
        body = self._request_body(payload, model_path, stream=False)
        data = self._post_json("/v1/chat/completions", body)
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")

    def stream(
        self,
        payload: dict[str, Any],
        model_path: str,
        *,
        should_stop,
    ) -> Iterator[StreamToken]:
        body = self._request_body(payload, model_path, stream=True)
        req = self._build_request("/v1/chat/completions", body)
        try:
            with urllib.request.urlopen(req, timeout=None) as response:
                for raw in response:
                    if should_stop():
                        break
                    text = raw.decode("utf-8", errors="replace").strip()
                    if not text or not text.startswith("data:"):
                        continue
                    event = text[5:].strip()
                    if event == "[DONE]":
                        break
                    try:
                        chunk = json.loads(event)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield StreamToken(str(content))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama is unavailable at {self.url}. "
                f"{sidecar_setup_hint(engine='ollama')}"
            ) from exc

    def release_external_memory(
        self, model_path: str | None = None
    ) -> tuple[bool, str | None]:
        """Best-effort Ollama model unload via keep_alive=0."""
        if not model_path:
            return False, "Ollama unload requires a model path"
        try:
            from seiso.inference.ollama_registry import (
                metadata_for_model_path,
                resolve_ollama_tag,
            )

            meta = metadata_for_model_path(model_path)
            tag = resolve_ollama_tag(model_path, meta)
        except Exception as exc:
            return False, str(exc)
        body = json.dumps({"model": tag, "keep_alive": 0}).encode("utf-8")
        target = urllib.parse.urljoin(f"{self.url}/", "api/generate")
        req = urllib.request.Request(
            target,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30):
                return True, None
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            return False, str(exc)

    def _resolve_model(self, model_path: str, payload: dict[str, Any]) -> str:
        from seiso.inference.ollama_registry import (
            ensure_model_registered,
            metadata_for_model_path,
        )

        meta = metadata_for_model_path(
            model_path, payload.get("model_metadata")
        )
        return ensure_model_registered(
            model_path,
            repo_id=meta.get("repo_id") if isinstance(meta.get("repo_id"), str) else None,
            metadata=meta,
            model_format=payload.get("model_format"),
        )

    def _request_body(
        self, payload: dict[str, Any], model_path: str, *, stream: bool
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._resolve_model(model_path, payload),
            "messages": payload.get("messages") or [],
            "max_tokens": int(payload.get("max_tokens", 512)),
            "temperature": float(payload.get("temperature", 0.0)),
            "stream": stream,
        }
        top_p = payload.get("top_p")
        if top_p is not None:
            body["top_p"] = float(top_p)
        tools = payload.get("tools_schemas")
        if tools:
            body["tools"] = tools
        return body

    def _build_request(self, path: str, body: dict[str, Any]) -> urllib.request.Request:
        target = urllib.parse.urljoin(f"{self.url}/", path.lstrip("/"))
        data = json.dumps(body).encode("utf-8")
        return urllib.request.Request(
            target,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = self._build_request(path, body)
        try:
            with urllib.request.urlopen(req, timeout=900) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama is unavailable at {self.url}. "
                f"{sidecar_setup_hint(engine='ollama')}"
            ) from exc
        return json.loads(raw)


class LlamaSwapClient:
    """Small OpenAI-compatible client for a local llama-swap sidecar."""

    def __init__(self, *, url: str | None = None, engine: str | None = None) -> None:
        self.url = (url or llamaswap_url()).rstrip("/")
        self.engine = engine or "llamacpp"

    def ensure_ready(self) -> None:
        """Verify llama-swap is reachable before preload."""
        if not llamaswap_health_ok(url=self.url):
            raise RuntimeError(
                f"llama-swap is not reachable at {self.url}. "
                f"{sidecar_setup_hint(url=self.url, engine=self.engine)}"
            )

    def complete(self, payload: dict[str, Any], model_path: str) -> str:
        body = self._request_body(payload, model_path, stream=False)
        data = self._post_json("/v1/chat/completions", body)
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")

    def stream(
        self,
        payload: dict[str, Any],
        model_path: str,
        *,
        should_stop,
    ) -> Iterator[StreamToken]:
        body = self._request_body(payload, model_path, stream=True)
        req = self._build_request("/v1/chat/completions", body)
        try:
            with urllib.request.urlopen(req, timeout=None) as response:
                for raw in response:
                    if should_stop():
                        break
                    text = raw.decode("utf-8", errors="replace").strip()
                    if not text or not text.startswith("data:"):
                        continue
                    event = text[5:].strip()
                    if event == "[DONE]":
                        break
                    try:
                        chunk = json.loads(event)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield StreamToken(str(content))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"llama-swap is unavailable at {self.url}. "
                f"{sidecar_setup_hint(url=self.url, engine=self.engine)}"
            ) from exc

    def _request_body(
        self, payload: dict[str, Any], model_path: str, *, stream: bool
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": llamaswap_model_name(model_path),
            "messages": payload.get("messages") or [],
            "max_tokens": int(payload.get("max_tokens", 512)),
            "temperature": float(payload.get("temperature", 0.0)),
            "stream": stream,
        }
        top_p = payload.get("top_p")
        if top_p is not None:
            body["top_p"] = float(top_p)
        tools = payload.get("tools_schemas")
        if tools:
            body["tools"] = tools
        return body

    def _build_request(self, path: str, body: dict[str, Any]) -> urllib.request.Request:
        target = urllib.parse.urljoin(f"{self.url}/", path.lstrip("/"))
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = env_str("SEISO_LLAMASWAP_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return urllib.request.Request(target, data=data, headers=headers, method="POST")

    def release_external_memory(
        self, model_path: str | None = None
    ) -> tuple[bool, str | None]:
        """Best-effort unload of llama-swap managed processes."""
        scope = env_str("SEISO_LLAMASWAP_UNLOAD_SCOPE", "all").strip().lower()
        if scope in {"0", "false", "none", "disabled"}:
            return False, "llama-swap unload disabled by SEISO_LLAMASWAP_UNLOAD_SCOPE"
        if scope == "model" and model_path:
            model = urllib.parse.quote(llamaswap_model_name(model_path), safe="")
            ok, reason = self._post_management(f"/api/models/unload/{model}")
            if ok:
                return True, None
        return self._post_management("/api/models/unload")

    def _post_management(self, path: str) -> tuple[bool, str | None]:
        target = urllib.parse.urljoin(f"{self.url}/", path.lstrip("/"))
        headers = {"Content-Type": "application/json"}
        api_key = env_str("SEISO_LLAMASWAP_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(
            target,
            data=b"{}",
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30):
                return True, None
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            return False, str(exc)

    def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = self._build_request(path, body)
        try:
            with urllib.request.urlopen(req, timeout=900) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"llama-swap is unavailable at {self.url}. "
                f"{sidecar_setup_hint(url=self.url, engine=self.engine)}"
            ) from exc
        return json.loads(raw)
