"""Forge server configuration — secure local-first defaults."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PrivateAttr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from forge.db.crypto import (
    generate_encryption_key,
    load_encryption_key_file,
    persist_encryption_key_file,
    resolve_encryption_key,
)
from seiso.security import generate_secret_key, resolve_data_dir

StorageMode = Literal["persistent", "ephemeral"]

# Local Forge + Vite dev — 127.0.0.1 and localhost are different browser origins.
DEFAULT_CORS_ORIGINS = "http://127.0.0.1:8765,http://localhost:8765,http://127.0.0.1:5173,http://localhost:5173"


class ForgeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SEISO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8765
    data_dir: Path = Field(default=Path("~/.seiso"))
    secret_key: str = ""
    allow_remote: bool = False
    trust_proxy: bool = False
    trusted_proxy_ips: str = ""
    secure_cookies: bool = False
    cors_origins: str = DEFAULT_CORS_ORIGINS
    hf_token: str = ""
    rate_limit: int = Field(default=120, ge=1)
    session_hours: int = Field(default=24, ge=1, le=168)
    debug: bool = False
    allow_compat_tools: bool = False
    allow_tools: bool = False
    allow_code_exec: bool = False
    inference_api_key: str = ""
    db_ephemeral: bool | None = None
    db_encryption_key: str = ""
    model_router_enabled: bool = False
    model_router_url: str = "http://127.0.0.1:8780"
    model_router_api_key: str = ""

    @field_validator("allow_compat_tools", mode="before")
    @classmethod
    def _legacy_allow_openai_tools_env(cls, value: object) -> object:
        """Accept SEISO_ALLOW_OPENAI_TOOLS when SEISO_ALLOW_COMPAT_TOOLS is unset."""
        if "SEISO_ALLOW_COMPAT_TOOLS" in os.environ:
            return value
        legacy = os.environ.get("SEISO_ALLOW_OPENAI_TOOLS")
        return legacy if legacy is not None else value

    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand_data_dir(cls, v: object) -> Path:
        return Path(str(v)).expanduser()

    _session_db_key: bytes | None = PrivateAttr(default=None)
    _storage_mode_configured: bool = PrivateAttr(default=False)

    def model_post_init(self, __context: object) -> None:
        self.data_dir = resolve_data_dir(self.data_dir)
        self._resolve_storage_mode()
        if not self.secret_key:
            key_file = self.data_dir / ".secret_key"
            if key_file.exists():
                self.secret_key = key_file.read_text().strip()
            else:
                self.secret_key = generate_secret_key()
                key_file.write_text(self.secret_key)
            with contextlib.suppress(OSError):
                key_file.chmod(0o600)
        # JWT signing material — refuse trivially short env/file values.
        if len(self.secret_key.encode("utf-8")) < 32:
            raise RuntimeError(
                "SEISO_SECRET_KEY (or data_dir/.secret_key) must be at least "
                "32 bytes — regenerate with a strong random value"
            )

        self._session_db_key = self._resolve_db_encryption_key()

        if not self.allow_remote:
            self.host = "127.0.0.1"

        if not self.inference_api_key:
            key_file = self.data_dir / ".inference_api_key"
            if key_file.exists():
                self.inference_api_key = key_file.read_text().strip()
            else:
                import secrets

                self.inference_api_key = f"seiso_sk_{secrets.token_urlsafe(32)}"
                key_file.write_text(self.inference_api_key)
                key_file.chmod(0o600)

        from forge.security.startup import validate_security_settings
        from forge.security.token_revocation import configure_revocation_store

        configure_revocation_store(self.data_dir)
        validate_security_settings(self)

    def rotate_inference_api_key(self) -> bool:
        """Regenerate Compat ``/v1`` key on disk.

        Returns False when ``SEISO_INFERENCE_API_KEY`` is env-bound (cannot
        rotate without changing the process environment).
        """
        if "SEISO_INFERENCE_API_KEY" in os.environ:
            return False
        import secrets

        key_file = self.data_dir / ".inference_api_key"
        key_file.parent.mkdir(parents=True, exist_ok=True)
        new_key = f"seiso_sk_{secrets.token_urlsafe(32)}"
        key_file.write_text(new_key, encoding="utf-8")
        key_file.chmod(0o600)
        self.inference_api_key = new_key
        return True

    @property
    def inference_api_key_owner_file(self) -> Path:
        return self.data_dir / ".inference_api_key.owner"

    def get_inference_api_key_owner(self) -> str | None:
        """Pubkey hex of the npub that owns the Compat ``/v1`` key, if bound."""
        path = self.inference_api_key_owner_file
        if not path.is_file():
            return None
        raw = path.read_text(encoding="utf-8").strip().lower()
        return raw if len(raw) == 64 and all(c in "0123456789abcdef" for c in raw) else None

    def bind_inference_api_key_owner(self, pubkey_hex: str) -> None:
        """Record that the Compat key belongs to this owner npub (pubkey hex)."""
        pubkey = pubkey_hex.strip().lower()
        if len(pubkey) != 64 or not all(c in "0123456789abcdef" for c in pubkey):
            raise ValueError("inference key owner must be a 64-char hex pubkey")
        path = self.inference_api_key_owner_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pubkey, encoding="utf-8")
        path.chmod(0o600)

    def clear_inference_api_key_owner(self) -> None:
        self.inference_api_key_owner_file.unlink(missing_ok=True)

    def sync_inference_api_key_owner(self, pubkey_hex: str) -> bool:
        """Bind Compat key to ``pubkey_hex``; rotate when the owner changes.

        Returns whether the key was rotated.

        When the Compat key is env-bound (``SEISO_INFERENCE_API_KEY``), rotation
        is impossible. First bind (no prior owner) is allowed; rebinding to a
        different owner is refused so a stale key cannot silently follow the
        next npub.
        """
        pubkey = pubkey_hex.strip().lower()
        current = self.get_inference_api_key_owner()
        if current == pubkey:
            return False
        if current is not None and "SEISO_INFERENCE_API_KEY" in os.environ:
            raise RuntimeError(
                "Cannot rebind Compat /v1 key owner while SEISO_INFERENCE_API_KEY "
                "is env-bound (rotation impossible). Unset the env var or rotate "
                "the key out-of-band before changing owners."
            )
        rotated = self.rotate_inference_api_key()
        self.bind_inference_api_key_owner(pubkey)
        return rotated

    def _resolve_storage_mode(self) -> None:
        marker = self.data_dir / ".storage_mode"
        env_configured = (
            "SEISO_DB_EPHEMERAL" in os.environ or "SEISO_DB_STORAGE_MODE" in os.environ
        )
        if env_configured:
            raw_mode = os.environ.get("SEISO_DB_STORAGE_MODE", "").strip().lower()
            if raw_mode:
                if raw_mode not in {"persistent", "ephemeral"}:
                    raise ValueError(
                        "SEISO_DB_STORAGE_MODE must be 'persistent' or 'ephemeral'"
                    )
                self.db_ephemeral = raw_mode == "ephemeral"
            elif self.db_ephemeral is None:
                self.db_ephemeral = False
            self._storage_mode_configured = True
            return
        if marker.exists():
            raw = marker.read_text(encoding="utf-8").strip().lower()
            if raw not in {"persistent", "ephemeral"}:
                raise ValueError(f"Invalid storage mode marker: {marker}")
            self.db_ephemeral = raw == "ephemeral"
            self._storage_mode_configured = True
            return
        # Before first onboarding, use an in-memory DB only to answer auth/status.
        self.db_ephemeral = True
        self._storage_mode_configured = False

    @property
    def storage_mode(self) -> StorageMode:
        return "ephemeral" if self.db_ephemeral else "persistent"

    @property
    def storage_mode_configured(self) -> bool:
        return self._storage_mode_configured

    def persist_storage_mode(self, mode: StorageMode) -> None:
        marker = self.data_dir / ".storage_mode"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{mode}\n", encoding="utf-8")
        marker.chmod(0o600)

    def _resolve_db_encryption_key(self) -> bytes:
        if self.db_encryption_key:
            return resolve_encryption_key(self.db_encryption_key)
        if self.db_ephemeral:
            return generate_encryption_key()
        key_file = self.data_dir / ".db_encryption_key"
        if key_file.exists():
            return load_encryption_key_file(key_file)
        key = generate_encryption_key()
        persist_encryption_key_file(key_file, key)
        return key

    def ensure_dirs(self) -> None:
        """Create data subdirectories once at startup."""
        for name in (
            "models",
            "checkpoints",
            "exports",
            "knowledge",
            "sandbox",
            "artifacts",
            "recipes",
            "uploads",
            "compress",
            "distill_rl",
            "hf_cache",
            "hf_tokens",
            "nostr_keys",
        ):
            (self.data_dir / name).mkdir(parents=True, exist_ok=True)

    @property
    def hf_cache_dir(self) -> Path:
        from seiso.models.hf_env import resolve_hf_cache_dir

        return resolve_hf_cache_dir(self.data_dir)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def trusted_proxy_ip_list(self) -> list[str]:
        return [ip.strip() for ip in self.trusted_proxy_ips.split(",") if ip.strip()]

    @property
    def cookie_secure(self) -> bool:
        """Set Secure on session/CSRF cookies (HTTPS reverse proxy or remote bind)."""
        return self.allow_remote or self.secure_cookies

    @property
    def rate_limit_enabled(self) -> bool:
        """Rate limits apply on all bindings (localhost included)."""
        return True

    @property
    def rate_limit_per_minute(self) -> int:
        """Stricter cap when exposed beyond loopback."""
        return self.rate_limit if self.allow_remote else max(self.rate_limit, 240)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "forge.db"

    @property
    def db_encryption_key_bytes(self) -> bytes:
        if self._session_db_key is None:
            raise RuntimeError("Database encryption key is not initialized")
        return self._session_db_key

    @property
    def hf_token_encryption_key(self) -> bytes:
        """Dedicated key for HF token files, stored separately from JWT secret."""
        key_file = self.data_dir / ".hf_token_encryption_key"
        if key_file.exists():
            return load_encryption_key_file(key_file)
        legacy = hashlib.sha256(f"seiso:hf-token:{self.secret_key}".encode()).digest()
        persist_encryption_key_file(key_file, legacy)
        return legacy

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def checkpoints_dir(self) -> Path:
        return self.data_dir / "checkpoints"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def knowledge_dir(self) -> Path:
        return self.data_dir / "knowledge"

    def write_runtime_config(self) -> None:
        """Persist non-secret runtime metadata for workers."""
        meta = {
            "data_dir": str(self.data_dir),
            "models_dir": str(self.models_dir),
            "checkpoints_dir": str(self.checkpoints_dir),
        }
        (self.data_dir / "runtime.json").write_text(json.dumps(meta, indent=2))


@lru_cache
def get_settings() -> ForgeSettings:
    return ForgeSettings()
