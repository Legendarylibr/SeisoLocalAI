"""Resolve Hugging Face credentials from user input, env, or CLI login cache."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from forge.db.crypto import decrypt_field, encrypt_field

TokenSource = Literal["request", "user_store", "env_seiso", "env_hf", "cli_cache", "none"]


@dataclass
class HfAuthStatus:
    cli_available: bool
    cli_binary: str | None
    cli_logged_in: bool
    token_configured: bool
    token_sources: list[str]


def _normalize_token(value: str | None) -> str | None:
    """Return a usable HF token, ignoring placeholders commonly left in .env files."""
    if not value:
        return None
    token = value.strip().strip("\"'")
    lowered = token.lower()
    if (
        not token
        or token.startswith("#")
        or lowered in {"none", "null", "optional", "your_token_here", "<your-token>"}
        or lowered.startswith("optional")
        or lowered.startswith("set ")
    ):
        return None
    return token


def find_hf_cli() -> str | None:
    """Return path to `hf` or legacy `huggingface-cli` if installed."""
    override = os.environ.get("HF_CLI", "").strip()
    if override and Path(override).exists():
        return override
    venv_bin = Path(sys.executable).parent
    for name in ("hf", "huggingface-cli"):
        candidate = venv_bin / name
        if candidate.exists():
            return str(candidate)
    for name in ("hf", "huggingface-cli"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _read_cli_token() -> str | None:
    """Read token saved by `huggingface-cli login` / `hf auth login`."""
    candidates = []
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        candidates.append(Path(hf_home).expanduser() / "token")
    home = Path.home()
    candidates.extend(
        [
            home / ".cache" / "huggingface" / "token",
            home / ".huggingface" / "token",
        ]
    )
    for path in candidates:
        if path.is_file():
            token = _normalize_token(path.read_text(encoding="utf-8"))
            if token:
                return token
    return None


def _user_token_path(data_dir: Path, user_id: str) -> Path:
    return data_dir / "hf_tokens" / user_id


def save_user_hf_token(data_dir: Path, user_id: str, token: str, *, encryption_key: bytes) -> None:
    path = _user_token_path(data_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encrypt_field(token.strip(), encryption_key), encoding="utf-8")
    path.chmod(0o600)


def clear_user_hf_token(data_dir: Path, user_id: str) -> None:
    path = _user_token_path(data_dir, user_id)
    if path.exists():
        path.unlink()


def load_user_hf_token(data_dir: Path, user_id: str, *, encryption_key: bytes) -> str | None:
    path = _user_token_path(data_dir, user_id)
    if not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    try:
        return _normalize_token(decrypt_field(raw, encryption_key))
    except Exception:
        return None


def resolve_hf_token(
    *,
    request_token: str | None = None,
    user_id: str | None = None,
    data_dir: Path | None = None,
    encryption_key: bytes | None = None,
    settings_token: str | None = None,
    prefer_cli: bool = False,
) -> tuple[str | None, TokenSource]:
    """Resolve an HF token. Returns (token, source)."""
    if prefer_cli:
        cli_token = _read_cli_token()
        if cli_token:
            return cli_token, "cli_cache"

    request_token = _normalize_token(request_token)
    if request_token:
        return request_token, "request"

    if user_id and data_dir and encryption_key:
        stored = load_user_hf_token(data_dir, user_id, encryption_key=encryption_key)
        if stored:
            return stored, "user_store"

    settings_token = _normalize_token(settings_token)
    if settings_token:
        return settings_token, "env_seiso"

    env_hf = _normalize_token(os.environ.get("HF_TOKEN")) or _normalize_token(
        os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if env_hf:
        return env_hf, "env_hf"

    cli_token = _read_cli_token()
    if cli_token:
        return cli_token, "cli_cache"

    return None, "none"


def resolve_hf_token_for_download(
    *,
    request_token: str | None = None,
    user_id: str | None = None,
    data_dir: Path | None = None,
    encryption_key: bytes | None = None,
    settings_token: str | None = None,
) -> tuple[str | None, TokenSource]:
    """Resolve an HF token for Hub downloads, dropping credentials that fail whoami."""
    token, source = resolve_hf_token(
        request_token=request_token,
        user_id=user_id,
        data_dir=data_dir,
        encryption_key=encryption_key,
        settings_token=settings_token,
    )
    if not token:
        return None, "none"

    from forge.services.hf_connectivity import probe_hf_hub

    result = probe_hf_hub(token=token)
    if result.token_valid:
        return token, source
    if not getattr(result, "reachable", True):
        return None, "none"
    if result.token_invalid or result.anonymous_ok:
        return None, "none"
    return token, source


def hf_auth_status(
    *,
    user_id: str | None = None,
    data_dir: Path | None = None,
    encryption_key: bytes | None = None,
    settings_token: str | None = None,
) -> HfAuthStatus:
    cli = find_hf_cli()
    cli_token = _read_cli_token()
    sources: list[str] = []

    token, source = resolve_hf_token(
        user_id=user_id,
        data_dir=data_dir,
        encryption_key=encryption_key,
        settings_token=settings_token,
    )
    if source != "none":
        sources.append(source)

    return HfAuthStatus(
        cli_available=cli is not None,
        cli_binary=cli,
        cli_logged_in=bool(cli_token),
        token_configured=token is not None,
        token_sources=sources,
    )
