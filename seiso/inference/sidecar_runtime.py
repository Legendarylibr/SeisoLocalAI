"""GGUF sidecar runtime: health probes, engine selection, and readiness status."""

from __future__ import annotations

import os
import platform
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from seiso.env import env_bool, env_str

DEFAULT_LLAMASWAP_URL = "http://127.0.0.1:8080"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class SidecarRuntime:
    available: bool
    url: str
    engine: str
    reason: str | None = None
    ollama_ready: bool = False
    llamaswap_ready: bool = False


# Backward-compatible alias used across the codebase.
LlamaSwapRuntime = SidecarRuntime


def _nvidia_visible() -> bool:
    try:
        from seiso.security.nvidia_boundary import nvidia_smi_visible

        return nvidia_smi_visible()
    except ImportError:
        return False


def ollama_url() -> str:
    raw = env_str("SEISO_OLLAMA_URL", DEFAULT_OLLAMA_URL).strip()
    return raw.rstrip("/") or DEFAULT_OLLAMA_URL


def ollama_cli_host(*, url: str | None = None) -> str:
    """Host:port for the Ollama CLI (OLLAMA_HOST) matching SEISO_OLLAMA_URL."""
    parsed = urllib.parse.urlparse(f"{(url or ollama_url()).rstrip('/')}/")
    if not parsed.hostname:
        return "127.0.0.1:11434"
    if parsed.port:
        return f"{parsed.hostname}:{parsed.port}"
    return parsed.hostname


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
        with urllib.request.urlopen(req, timeout=_ollama_health_timeout_s()) as response:
            return 200 <= int(getattr(response, "status", 200)) < 300
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def ollama_registration_available() -> bool:
    """True when the Ollama CLI can register models (independent of chat engine)."""
    return bool(shutil.which("ollama")) and ollama_health_ok()


def preferred_sidecar_engine() -> str:
    """Choose the isolated GGUF engine for this machine."""
    override = env_str("SEISO_LLAMASWAP_ENGINE", "").strip().lower()
    if override and override != "auto":
        return override
    if platform.system() == "Darwin":
        return "llamacpp"
    if _nvidia_visible() and ollama_health_ok():
        return "ollama"
    return "llamacpp"


def preferred_llamaswap_engine() -> str:
    """Backward-compatible alias for preferred_sidecar_engine."""
    return preferred_sidecar_engine()


def llamaswap_url() -> str:
    raw = env_str("SEISO_LLAMASWAP_URL", DEFAULT_LLAMASWAP_URL).strip()
    return raw.rstrip("/") or DEFAULT_LLAMASWAP_URL


def sidecar_enabled() -> bool:
    if "SEISO_LLAMASWAP_ENABLED" in os.environ:
        return env_bool("SEISO_LLAMASWAP_ENABLED", False)
    return bool(
        os.environ.get("SEISO_LLAMASWAP_URL") or shutil.which("llama-swap") or ollama_health_ok()
    )


def llamaswap_enabled() -> bool:
    """Backward-compatible alias for sidecar_enabled."""
    return sidecar_enabled()


def sidecar_setup_hint(*, url: str | None = None, engine: str | None = None) -> str:
    """Actionable setup text for the native Linux sidecar path."""
    target = (url or llamaswap_url()).rstrip("/")
    selected = engine or preferred_sidecar_engine()
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


def llamaswap_setup_hint(*, url: str | None = None, engine: str | None = None) -> str:
    return sidecar_setup_hint(url=url, engine=engine)


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
        with urllib.request.urlopen(req, timeout=_llamaswap_health_timeout_s()) as response:
            return 200 <= int(getattr(response, "status", 200)) < 300
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def sidecar_stack_ready() -> tuple[bool, bool]:
    """Return (ollama_ready, llamaswap_ready)."""
    return ollama_health_ok(), llamaswap_health_ok()


def sidecar_status() -> SidecarRuntime:
    """Report isolated GGUF sidecar readiness and active engine."""
    url = llamaswap_url()
    engine = preferred_sidecar_engine()
    ollama_ready, swap_ready = sidecar_stack_ready()

    if not sidecar_enabled():
        return SidecarRuntime(
            available=False,
            url=url,
            engine=engine,
            reason=sidecar_setup_hint(url=url, engine=engine),
            ollama_ready=ollama_ready,
            llamaswap_ready=swap_ready,
        )

    if engine == "ollama" and ollama_ready:
        return SidecarRuntime(
            available=True,
            url=ollama_url(),
            engine="ollama",
            ollama_ready=True,
            llamaswap_ready=swap_ready,
        )

    if swap_ready:
        return SidecarRuntime(
            available=True,
            url=url,
            engine="llamacpp",
            ollama_ready=ollama_ready,
            llamaswap_ready=True,
        )

    return SidecarRuntime(
        available=False,
        url=url,
        engine=engine,
        reason=(
            f"Neither Ollama ({ollama_url()}) nor llama-swap ({url}) is reachable. "
            f"{sidecar_setup_hint(url=url, engine=engine)}"
        ),
        ollama_ready=ollama_ready,
        llamaswap_ready=False,
    )


def llamaswap_status() -> SidecarRuntime:
    """Backward-compatible alias for sidecar_status."""
    return sidecar_status()
