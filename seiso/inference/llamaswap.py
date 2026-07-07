"""Isolated GGUF sidecar integration (Ollama-first, llama-swap fallback)."""

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
from typing import Any, Protocol

from seiso.env import env_bool, env_str
from seiso.inference.streaming import StreamToken

DEFAULT_LLAMASWAP_URL = "http://127.0.0.1:8080"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class LlamaSwapRuntime:
    available: bool
    url: str
    engine: str
    reason: str | None = None
    ollama_ready: bool = False
    llamaswap_ready: bool = False


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


def _nvidia_visible() -> bool:
    try:
        from seiso.security.nvidia_boundary import nvidia_smi_visible

        return nvidia_smi_visible()
    except ImportError:
        return False


def ollama_url() -> str:
    raw = env_str("SEISO_OLLAMA_URL", DEFAULT_OLLAMA_URL).strip()
    return raw.rstrip("/") or DEFAULT_OLLAMA_URL


def _ollama_health_timeout_s() -> float:
    raw = env_str("SEISO_OLLAMA_HEALTH_TIMEOUT_S", "0.35").strip()
    try:
        return max(0.05, float(raw))
    except ValueError:
        return 0.35


def ollama_health_ok(*, url: str | None = None) -> bool:
    """Return True when Ollama's local API is reachable."""
    target = urllib.parse.urljoin(f"{(url or ollama_url()).rstrip('/')}/", "api/tags")
    req = urllib.request.Request(target, method="GET")
    try:
        with urllib.request.urlopen(
            req, timeout=_ollama_health_timeout_s()
        ) as response:
            return 200 <= int(getattr(response, "status", 200)) < 300
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def preferred_llamaswap_engine() -> str:
    """Choose the isolated GGUF engine for this machine."""
    override = env_str("SEISO_LLAMASWAP_ENGINE", "").strip().lower()
    if override and override != "auto":
        return override
    if platform.system() == "Darwin":
        return "llamacpp"
    if _nvidia_visible() and ollama_health_ok():
        return "ollama"
    return "llamacpp"


def llamaswap_url() -> str:
    raw = env_str("SEISO_LLAMASWAP_URL", DEFAULT_LLAMASWAP_URL).strip()
    return raw.rstrip("/") or DEFAULT_LLAMASWAP_URL


def llamaswap_enabled() -> bool:
    if "SEISO_LLAMASWAP_ENABLED" in os.environ:
        return env_bool("SEISO_LLAMASWAP_ENABLED", False)
    return bool(
        os.environ.get("SEISO_LLAMASWAP_URL")
        or shutil.which("llama-swap")
        or ollama_health_ok()
    )


def llamaswap_setup_hint(*, url: str | None = None, engine: str | None = None) -> str:
    """Actionable setup text for the native Linux sidecar path."""
    target = (url or llamaswap_url()).rstrip("/")
    selected = engine or preferred_llamaswap_engine()
    ollama_target = ollama_url()
    if selected == "ollama":
        engine_hint = (
            f"Install/start Ollama at {ollama_target} "
            "(curl -fsSL https://ollama.com/install.sh | sh && ollama serve)."
        )
    else:
        engine_hint = (
            f"Start Ollama at {ollama_target} for the preferred path, or configure "
            f"llama-swap at {target} for llama.cpp fallback."
        )
    return (
        f"{engine_hint} Optional llama-swap fallback: {target}. "
        "Set SEISO_LLAMA_ALLOW_INPROCESS_NATIVE_LINUX=1 only "
        "if you accept that in-process llama.cpp can crash Forge."
    )


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


def sidecar_stack_ready() -> tuple[bool, bool]:
    """Return (ollama_ready, llamaswap_ready)."""
    return ollama_health_ok(), llamaswap_health_ok()


def llamaswap_status() -> LlamaSwapRuntime:
    url = llamaswap_url()
    engine = preferred_llamaswap_engine()
    ollama_ready, swap_ready = sidecar_stack_ready()

    if not llamaswap_enabled():
        return LlamaSwapRuntime(
            available=False,
            url=url,
            engine=engine,
            reason=llamaswap_setup_hint(url=url, engine=engine),
            ollama_ready=ollama_ready,
            llamaswap_ready=swap_ready,
        )

    if engine == "ollama" and ollama_ready:
        return LlamaSwapRuntime(
            available=True,
            url=ollama_url(),
            engine="ollama",
            ollama_ready=True,
            llamaswap_ready=swap_ready,
        )

    if swap_ready:
        return LlamaSwapRuntime(
            available=True,
            url=url,
            engine="llamacpp",
            ollama_ready=ollama_ready,
            llamaswap_ready=True,
        )

    return LlamaSwapRuntime(
        available=False,
        url=url,
        engine=engine,
        reason=(
            f"Neither Ollama ({ollama_url()}) nor llama-swap ({url}) is reachable. "
            f"{llamaswap_setup_hint(url=url, engine=engine)}"
        ),
        ollama_ready=ollama_ready,
        llamaswap_ready=False,
    )


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
    selected = engine or preferred_llamaswap_engine()
    if selected == "ollama" and ollama_health_ok():
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
                f"{llamaswap_setup_hint(engine='ollama')}"
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
                f"{llamaswap_setup_hint(engine='ollama')}"
            ) from exc

    def release_external_memory(
        self, model_path: str | None = None
    ) -> tuple[bool, str | None]:
        """Best-effort Ollama model unload via keep_alive=0."""
        if not model_path:
            return False, "Ollama unload requires a model path"
        try:
            from forge.services.ollama_registry import (
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
        from forge.services.ollama_registry import (
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
                f"{llamaswap_setup_hint(engine='ollama')}"
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
                f"{llamaswap_setup_hint(url=self.url, engine=self.engine)}"
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
                f"{llamaswap_setup_hint(url=self.url, engine=self.engine)}"
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
                f"{llamaswap_setup_hint(url=self.url, engine=self.engine)}"
            ) from exc
        return json.loads(raw)
