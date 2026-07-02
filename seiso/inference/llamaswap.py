"""llama-swap sidecar integration for local GGUF chat."""

from __future__ import annotations

import json
import os
import platform
import shutil
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seiso.env import env_bool, env_str
from seiso.inference.streaming import StreamToken

DEFAULT_LLAMASWAP_URL = "http://127.0.0.1:8080"


@dataclass(frozen=True)
class LlamaSwapRuntime:
    available: bool
    url: str
    engine: str
    reason: str | None = None


def _nvidia_visible() -> bool:
    try:
        from seiso.security.nvidia_boundary import nvidia_smi_visible

        return nvidia_smi_visible()
    except ImportError:
        return False


def preferred_llamaswap_engine() -> str:
    """Choose the llama-swap managed engine for this machine."""
    override = env_str("SEISO_LLAMASWAP_ENGINE", "").strip().lower()
    if override and override != "auto":
        return override
    if platform.system() == "Darwin":
        return "llamacpp"
    if _nvidia_visible():
        return "ollama"
    return "llamacpp"


def llamaswap_url() -> str:
    raw = env_str("SEISO_LLAMASWAP_URL", DEFAULT_LLAMASWAP_URL).strip()
    return raw.rstrip("/") or DEFAULT_LLAMASWAP_URL


def llamaswap_enabled() -> bool:
    if "SEISO_LLAMASWAP_ENABLED" in os.environ:
        return env_bool("SEISO_LLAMASWAP_ENABLED", False)
    return bool(os.environ.get("SEISO_LLAMASWAP_URL") or shutil.which("llama-swap"))


def _llamaswap_health_timeout_s() -> float:
    raw = env_str("SEISO_LLAMASWAP_HEALTH_TIMEOUT_S", "0.5").strip()
    try:
        return max(0.05, float(raw))
    except ValueError:
        return 0.5


def llamaswap_health_ok(*, url: str | None = None) -> bool:
    """Return True when the configured llama-swap sidecar is reachable."""
    target = urllib.parse.urljoin(f"{(url or llamaswap_url()).rstrip('/')}/", "health")
    req = urllib.request.Request(target, method="GET")
    try:
        with urllib.request.urlopen(
            req, timeout=_llamaswap_health_timeout_s()
        ) as response:
            return 200 <= int(getattr(response, "status", 200)) < 300
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def llamaswap_status() -> LlamaSwapRuntime:
    url = llamaswap_url()
    engine = preferred_llamaswap_engine()
    if not llamaswap_enabled():
        return LlamaSwapRuntime(
            available=False,
            url=url,
            engine=engine,
            reason="Set SEISO_LLAMASWAP_URL or install llama-swap to enable it.",
        )
    if not llamaswap_health_ok(url=url):
        return LlamaSwapRuntime(
            available=False,
            url=url,
            engine=engine,
            reason=f"llama-swap is configured but not reachable at {url}. Start the sidecar.",
        )
    return LlamaSwapRuntime(
        available=True,
        url=url,
        engine=engine,
    )


def llamaswap_model_name(model_path: str) -> str:
    """Resolve the model identifier expected by the llama-swap OpenAI API."""
    override = env_str("SEISO_LLAMASWAP_MODEL", "").strip()
    if override:
        return override
    return str(Path(model_path).expanduser())


class LlamaSwapClient:
    """Small OpenAI-compatible client for a local llama-swap sidecar."""

    def __init__(self, *, url: str | None = None, engine: str | None = None) -> None:
        self.url = (url or llamaswap_url()).rstrip("/")
        self.engine = engine or preferred_llamaswap_engine()

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
                f"llama-swap is unavailable at {self.url}. Start llama-swap or switch to llama.cpp."
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
        headers = {
            "Content-Type": "application/json",
            "X-Seiso-LlamaSwap-Engine": self.engine,
        }
        api_key = env_str("SEISO_LLAMASWAP_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return urllib.request.Request(target, data=data, headers=headers, method="POST")

    def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = self._build_request(path, body)
        try:
            with urllib.request.urlopen(req, timeout=900) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"llama-swap is unavailable at {self.url}. Start llama-swap or switch to llama.cpp."
            ) from exc
        return json.loads(raw)
