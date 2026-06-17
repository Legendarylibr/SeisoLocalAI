"""Forge server configuration — secure local-first defaults."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field, PrivateAttr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from seiso.security import generate_secret_key, resolve_data_dir

from forge.db.crypto import generate_encryption_key, resolve_encryption_key


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
    cors_origins: str = "http://127.0.0.1:8765,http://localhost:5173"
    hf_token: str = ""
    rate_limit: int = Field(default=120, ge=1)
    session_hours: int = Field(default=24, ge=1, le=168)
    debug: bool = False
    allow_openai_tools: bool = False
    allow_tools: bool = False
    allow_code_exec: bool = False
    ollama_base_url: str = "http://127.0.0.1:11434"
    db_ephemeral: bool = True
    db_encryption_key: str = ""
    autodefense_enabled: bool = False
    autodefense_url: str = "http://127.0.0.1:8000"
    autodefense_api_key: str = ""
    autodefense_timeout: float = Field(default=10.0, ge=1.0, le=120.0)
    autodefense_fail_open: bool = True

    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand_data_dir(cls, v: object) -> Path:
        return Path(str(v)).expanduser()

    _session_db_key: bytes | None = PrivateAttr(default=None)

    def model_post_init(self, __context: object) -> None:
        self.data_dir = resolve_data_dir(self.data_dir)
        if not self.secret_key:
            key_file = self.data_dir / ".secret_key"
            if key_file.exists():
                self.secret_key = key_file.read_text().strip()
            else:
                self.secret_key = generate_secret_key()
                key_file.write_text(self.secret_key)
                key_file.chmod(0o600)

        self._session_db_key = self._resolve_db_encryption_key()

        if self.db_ephemeral:
            legacy_db = self.data_dir / "forge.db"
            if legacy_db.exists():
                legacy_db.unlink()

        if not self.allow_remote:
            self.host = "127.0.0.1"

    def _resolve_db_encryption_key(self) -> bytes:
        if self.db_encryption_key:
            return resolve_encryption_key(self.db_encryption_key)
        if self.db_ephemeral:
            return generate_encryption_key()
        key_file = self.data_dir / ".db_encryption_key"
        if key_file.exists():
            return resolve_encryption_key(key_file.read_text().strip())
        key = generate_encryption_key()
        key_file.write_bytes(key)
        key_file.chmod(0o600)
        return key

    def ensure_dirs(self) -> None:
        """Create data subdirectories once at startup."""
        for name in ("models", "checkpoints", "exports", "knowledge", "sandbox", "artifacts", "recipes", "uploads", "rl_quant", "compress", "image_compress"):
            (self.data_dir / name).mkdir(parents=True, exist_ok=True)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "forge.db"

    @property
    def db_encryption_key_bytes(self) -> bytes:
        if self._session_db_key is None:
            raise RuntimeError("Database encryption key is not initialized")
        return self._session_db_key

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
