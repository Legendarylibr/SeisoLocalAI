"""Hugging Face Hub connectivity, auth validation, and inference runtime checks."""

from __future__ import annotations

import time
from importlib.util import find_spec
from dataclasses import dataclass, field
from typing import Any

from forge.services.hf_auth import hf_auth_status, resolve_hf_token
from seiso.models.hf_env import hf_transfer_stack, resolve_hf_cache_dir


@dataclass
class HfConnectivityResult:
    """Result of probing huggingface.co reachability."""

    reachable: bool
    latency_ms: int | None = None
    token_valid: bool = False
    token_invalid: bool = False
    token_username: str | None = None
    anonymous_ok: bool = False
    error: str | None = None
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reachable": self.reachable,
            "latency_ms": self.latency_ms,
            "token_valid": self.token_valid,
            "token_invalid": self.token_invalid,
            "token_username": self.token_username,
            "anonymous_ok": self.anonymous_ok,
            "error": self.error,
            "warning": self.warning,
        }


@dataclass
class InferenceRuntimeStatus:
    """Local inference engine dependencies required to load models."""

    llamacpp: bool = False
    mlx: bool = False
    torch: bool = False
    huggingface_hub: bool = False
    install_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "llamacpp": self.llamacpp,
            "mlx": self.mlx,
            "torch": self.torch,
            "huggingface_hub": self.huggingface_hub,
            "install_hints": self.install_hints,
        }


def _dep_status(module: str) -> bool:
    try:
        return find_spec(module) is not None
    except Exception:
        return False


def check_inference_runtime() -> InferenceRuntimeStatus:
    """Report which local inference stacks are importable."""
    status = InferenceRuntimeStatus(
        llamacpp=_dep_status("llama_cpp"),
        mlx=_dep_status("mlx_lm"),
        torch=_dep_status("torch"),
        huggingface_hub=_dep_status("huggingface_hub"),
    )
    hints: list[str] = []
    if not status.huggingface_hub:
        hints.append('pip install -e ".[forge]"  # includes huggingface-hub')
    if not status.llamacpp:
        hints.append('pip install -e ".[llamacpp]"  # GGUF chat via llama.cpp')
    if not status.mlx and not status.torch:
        hints.append('pip install -e ".[mlx]" or ".[train]"  # safetensors inference')
    status.install_hints = hints
    return status


def _probe_hf_hub_anonymous(api: Any, *, timeout: float, started: float) -> HfConnectivityResult:
    """Anonymous reachability check — public models work without credentials."""
    try:
        api.model_info("gpt2", timeout=timeout)
        latency_ms = int((time.monotonic() - started) * 1000)
        return HfConnectivityResult(
            reachable=True,
            latency_ms=latency_ms,
            anonymous_ok=True,
        )
    except Exception as exc:
        return HfConnectivityResult(reachable=False, error=_format_hub_error(exc))


def probe_hf_hub(*, token: str | None = None, timeout: float = 15.0) -> HfConnectivityResult:
    """
    Verify Hub reachability using the standard HfApi probe.

    With a token, validates credentials via whoami(). When the token is invalid,
    still probes anonymous access so public model downloads can proceed.
    """
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        return HfConnectivityResult(
            reachable=False,
            error="huggingface_hub is not installed — run: pip install -e '.[forge]'",
        )

    api = HfApi()
    started = time.monotonic()
    token_warning: str | None = None
    token_invalid = False

    if token:
        try:
            who = api.whoami(token=token)
            latency_ms = int((time.monotonic() - started) * 1000)
            return HfConnectivityResult(
                reachable=True,
                latency_ms=latency_ms,
                token_valid=True,
                token_username=who.get("name") if isinstance(who, dict) else None,
                anonymous_ok=True,
            )
        except HfHubHTTPError as exc:
            if exc.response is not None and exc.response.status_code in (401, 403):
                token_invalid = True
                token_warning = (
                    "Saved Hugging Face token was rejected — public downloads still work, "
                    "but gated models need a valid token in Settings or `hf auth login`."
                )
            else:
                return HfConnectivityResult(
                    reachable=False,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    error=_format_hub_error(exc),
                )
        except Exception as exc:
            return HfConnectivityResult(reachable=False, error=_format_hub_error(exc))

    anon = _probe_hf_hub_anonymous(api, timeout=timeout, started=started)
    if token_invalid:
        anon.token_invalid = True
        anon.warning = token_warning
        if anon.reachable and anon.anonymous_ok:
            anon.error = None
    return anon


def _format_hub_error(exc: Exception) -> str:
    msg = str(exc).strip() or exc.__class__.__name__
    lowered = msg.lower()
    if "connection" in lowered or "network" in lowered or "resolve" in lowered:
        return f"Cannot reach huggingface.co — check your network connection. ({msg})"
    if "timeout" in lowered or "timed out" in lowered:
        return f"Hugging Face Hub timed out — try again or increase HF_HUB_DOWNLOAD_TIMEOUT. ({msg})"
    return msg


def build_hf_status(
    *,
    user_id: str | None = None,
    data_dir: Any | None = None,
    encryption_key: bytes | None = None,
    settings_token: str | None = None,
    probe: bool = True,
) -> dict[str, Any]:
    """Full Hub status for Settings / Hub UI."""
    auth = hf_auth_status(
        user_id=user_id,
        data_dir=data_dir,
        encryption_key=encryption_key,
        settings_token=settings_token,
    )
    token, token_source = resolve_hf_token(
        user_id=user_id,
        data_dir=data_dir,
        encryption_key=encryption_key,
        settings_token=settings_token,
    )
    transfer = hf_transfer_stack()
    connectivity = probe_hf_hub(token=token) if probe else HfConnectivityResult(reachable=False)
    runtime = check_inference_runtime()

    ready_for_download = (
        connectivity.reachable
        and runtime.huggingface_hub
        and (connectivity.anonymous_ok or connectivity.token_valid)
    )
    ready_for_gguf_chat = ready_for_download and runtime.llamacpp
    ready_for_local_chat = ready_for_download and (
        runtime.llamacpp or runtime.mlx or runtime.torch
    )

    return {
        "auth": {
            "cli_available": auth.cli_available,
            "cli_binary": auth.cli_binary,
            "cli_logged_in": auth.cli_logged_in,
            "token_configured": auth.token_configured,
            "token_sources": auth.token_sources,
            "token_source": token_source,
            "token_invalid": connectivity.token_invalid,
        },
        "connectivity": connectivity.to_dict(),
        "transfer": transfer,
        "cache_dir": str(resolve_hf_cache_dir(data_dir)),
        "runtime": {
            "llamacpp": runtime.llamacpp,
            "mlx": runtime.mlx,
            "torch": runtime.torch,
            "huggingface_hub": runtime.huggingface_hub,
            "install_hints": runtime.install_hints,
        },
        "ready_for_download": ready_for_download,
        "ready_for_gguf_chat": ready_for_gguf_chat,
        "ready_for_local_chat": ready_for_local_chat,
    }


def assert_hub_ready_for_download(
    *,
    user_id: str | None = None,
    data_dir: Any | None = None,
    encryption_key: bytes | None = None,
    settings_token: str | None = None,
) -> None:
    """Raise ValueError with actionable guidance when Hub downloads cannot proceed."""
    runtime = check_inference_runtime()
    if not runtime.huggingface_hub:
        raise ValueError(
            "huggingface_hub is not installed. Install Seiso with: pip install -e '.[forge,llamacpp]'"
        )

    token, _ = resolve_hf_token(
        user_id=user_id,
        data_dir=data_dir,
        encryption_key=encryption_key,
        settings_token=settings_token,
    )
    result = probe_hf_hub(token=token)
    if not result.reachable:
        raise ValueError(result.error or "Cannot reach Hugging Face Hub")
    if not result.anonymous_ok and not result.token_valid:
        raise ValueError(
            result.error
            or "Cannot reach Hugging Face Hub — check your network connection"
        )
