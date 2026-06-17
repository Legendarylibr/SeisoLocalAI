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
    candidates.append(Path.home() / ".cache" / "huggingface" / "token")
    for path in candidates:
        if path.is_file():
            token = path.read_text(encoding="utf-8").strip()
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
        return decrypt_field(raw, encryption_key).strip()
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

    if request_token and request_token.strip():
        return request_token.strip(), "request"

    if user_id and data_dir and encryption_key:
        stored = load_user_hf_token(data_dir, user_id, encryption_key=encryption_key)
        if stored:
            return stored, "user_store"

    if settings_token and settings_token.strip():
        return settings_token.strip(), "env_seiso"

    env_hf = os.environ.get("HF_TOKEN", "").strip() or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
    if env_hf:
        return env_hf, "env_hf"

    cli_token = _read_cli_token()
    if cli_token:
        return cli_token, "cli_cache"

    return None, "none"


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

    user_has = bool(
        user_id
        and data_dir
        and encryption_key
        and load_user_hf_token(data_dir, user_id, encryption_key=encryption_key)
    )
    if user_has and "user_store" not in sources:
        sources.append("user_store")

    return HfAuthStatus(
        cli_available=cli is not None,
        cli_binary=cli,
        cli_logged_in=bool(cli_token),
        token_configured=token is not None,
        token_sources=sources,
    )
