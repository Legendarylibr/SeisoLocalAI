"""Router configuration loaded from YAML + environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RouterSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SEISO_ROUTER_", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8780
    mode: str = "local"  # local | prod

    config_path: Path = Field(default=Path("deploy/model-router/config/router.local.yaml"))

    llamaswap_url: str = ""
    specialists_path: Path = Field(default=Path("deploy/model-router/config/specialists.json"))
    policy_state_path: Path = Field(default=Path("data/router/policy_state.json"))

    hardware: str = "gpu"
    max_vram_hot: int = 2
    default_idle_sleep_sec: int = 300
    lifecycle_poll_sec: float = 15.0
    wake_timeout_sec: float = 120.0
    request_timeout_sec: float = 300.0

    enable_rl_policy: bool = True
    rl_ucb_c: float = 1.5
    rl_prior_weight: float = 4.0
    rl_warmup_pulls: int = 3
    rl_seed: int = 13

    # Prod
    api_keys: list[str] = Field(default_factory=list)
    rate_limit_rpm: int = 0  # 0 = unlimited
    rate_limit_burst: int = 20
    log_json: bool = False
    trust_proxy: bool = False

    fallback_route_id: str = "general"
    allow_explicit_model: bool = True

    @field_validator("api_keys", mode="before")
    @classmethod
    def _parse_api_keys(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @classmethod
    def load(cls, path: Path | None = None, **overrides: Any) -> "RouterSettings":
        if path and path.is_file():
            return cls.from_yaml(path, **overrides)
        return cls(**overrides)

    @classmethod
    def from_yaml(cls, path: Path, **overrides: Any) -> RouterSettings:
        data: dict[str, Any] = {}
        if path.is_file():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(raw, dict):
                data = raw
        merged = {**data, **overrides}
        return cls(**merged)


def resolve_paths(settings: RouterSettings, base: Path | None = None) -> RouterSettings:
    """Resolve relative paths against repo root or given base."""
    root = base or Path.cwd()
    updates: dict[str, Any] = {}
    for name in ("config_path", "specialists_path", "policy_state_path"):
        p = getattr(settings, name)
        if not p.is_absolute():
            updates[name] = root / p
    if updates:
        return settings.model_copy(update=updates)
    return settings
