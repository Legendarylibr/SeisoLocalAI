from __future__ import annotations

import argparse
import os
import secrets
import sys
from typing import Any

from adaptive_quant.cli.startup_overrides import (
    merge_override,
    parse_config_override,
    privileged_override_keys,
)
from adaptive_quant.configuration.validation import validate_cli_path_argument
from adaptive_quant.logging_utils import enforce_safe_parsed_json, safe_json_loads
from adaptive_quant.ui.rl_fields import collect_rl_set_overrides

_LAUNCHER_TOKEN_HEADER = "X-Launcher-Token"
_MAX_API_BODY_BYTES = 1 << 20
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_ALLOW_REMOTE_ENV = "ADAPTIVE_RL_LAUNCHER_ALLOW_REMOTE"


def launcher_token_header() -> str:
    return _LAUNCHER_TOKEN_HEADER


def generate_launcher_token() -> str:
    return secrets.token_urlsafe(32)


def validate_bind_host(host: str) -> None:
    """Refuse non-loopback binds unless explicitly opted in (local-first default)."""
    normalized = host.strip().lower()
    if normalized in _LOOPBACK_HOSTS:
        return
    if os.environ.get(_ALLOW_REMOTE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    raise SystemExit(
        f"Refusing to bind launcher to {host!r}. "
        "Use 127.0.0.1 for local use, or set "
        f"{_ALLOW_REMOTE_ENV}=1 to allow remote bind addresses."
    )


def read_api_json_body(headers: Any, body_stream: Any) -> dict[str, Any]:
    """Read and parse a bounded JSON object from an HTTP POST body."""
    try:
        length = int(headers.get("Content-Length", "0"))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid Content-Length") from exc
    if length <= 0:
        return {}
    if length > _MAX_API_BODY_BYTES:
        raise ValueError(f"request body exceeds {_MAX_API_BODY_BYTES} bytes")
    raw = body_stream.read(length)
    if len(raw) != length:
        raise ValueError("short request body")
    parsed = safe_json_loads(raw.decode("utf-8"), label="launcher API body")
    enforce_safe_parsed_json(parsed, label="launcher API body")
    if not isinstance(parsed, dict):
        raise ValueError("JSON body must be an object")
    return parsed


def verify_launcher_token(headers: Any, expected: str) -> None:
    provided = str(headers.get(_LAUNCHER_TOKEN_HEADER, "")).strip()
    if not provided or not secrets.compare_digest(provided, expected):
        raise PermissionError("invalid or missing launcher token")


def audit_log(message: str) -> None:
    print(f"[launcher-audit] {message}", file=sys.stderr, flush=True)


def _override_lines_from_options(options: dict[str, Any]) -> list[str]:
    lines = list(collect_rl_set_overrides(options))
    raw = options.get("config_overrides") or []
    if isinstance(raw, str):
        lines.extend(line.strip() for line in raw.splitlines() if line.strip())
    elif isinstance(raw, list):
        lines.extend(str(item).strip() for item in raw if str(item).strip())
    return lines


def gather_overrides_from_options(options: dict[str, Any]) -> dict[str, Any]:
    """Merge structured UI options and config_overrides into a startup override dict."""
    overrides: dict[str, Any] = {}
    for line in _override_lines_from_options(options):
        try:
            key, value = parse_config_override(line)
        except argparse.ArgumentTypeError as exc:
            raise ValueError(str(exc)) from exc
        merge_override(overrides, key, value)
    return overrides


def validate_run_options(options: dict[str, Any]) -> None:
    """Server-side validation mirroring CLI override rules before preview/run."""
    config = str(options.get("config", "")).strip()
    if config:
        validate_cli_path_argument("config", config)

    overrides = gather_overrides_from_options(options)
    blocked = privileged_override_keys(overrides)
    if blocked and not options.get("privileged_overrides"):
        keys = ", ".join(blocked)
        raise ValueError(
            "Privileged configuration changes require enabling "
            f"'Allow privileged --set overrides' in environment settings: {keys}"
        )

    smi_path = str(options.get("nvidia_smi_path", "")).strip()
    if smi_path:
        validate_cli_path_argument("nvidia_smi_path", smi_path)


__all__ = [
    "audit_log",
    "gather_overrides_from_options",
    "generate_launcher_token",
    "launcher_token_header",
    "read_api_json_body",
    "validate_bind_host",
    "validate_run_options",
    "verify_launcher_token",
]
