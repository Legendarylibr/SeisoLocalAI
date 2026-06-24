"""Router configuration loaded from YAML + environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RouterSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SEISO_ROUTER_", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8780
    mode: str = "local"  # local | prod
    inference_backend: str = "llamacpp"  # llamacpp | vllm (local stack selector)
    vllm_sleep_mode: bool = False  # required for Nemotron orchestrator routing

    config_path: Path = Field(default=Path("deploy/model-router/config/router.local.yaml"))

    llamaswap_url: str = ""
    specialists_path: Path = Field(
        default=Path("deploy/model-router/config/specialists.local.llamacpp.json")
    )
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

    # Nemotron-Orchestrator-8B (ToolOrchestra-style routing)
    routing_mode: str = "heuristic"  # heuristic | nemotron
    orchestrator_url: str = ""
    orchestrator_model: str = "seiso-orchestrator"
    orchestrator_timeout_sec: float = 120.0
    orchestrator_temperature: float = 0.7
    orchestrator_max_tokens: int = 512

    # LiteLLM executes all vLLM-stack completions (local + cloud catalog routes)
    litellm_routing_strategy: str = "simple-shuffle"

    @model_validator(mode="after")
    def _nemotron_requires_vllm_sleep(self) -> RouterSettings:
        if self.routing_mode.strip().lower() != "nemotron":
            return self
        issues: list[str] = []
        if self.inference_backend.strip().lower() != "vllm":
            issues.append("inference_backend must be vllm")
        if not self.vllm_sleep_mode:
            issues.append("vllm_sleep_mode must be true")
        if not self.orchestrator_url.strip():
            issues.append("orchestrator_url must be set")
        if issues:
            raise ValueError(
                "routing_mode=nemotron is only supported with vLLM sleep mode: " + "; ".join(issues)
            )
        return self

    def nemotron_orchestrator_enabled(self) -> bool:
        """True when Nemotron routing is configured for a vLLM sleep-mode stack."""
        return (
            self.routing_mode.strip().lower() == "nemotron"
            and self.inference_backend.strip().lower() == "vllm"
            and self.vllm_sleep_mode
            and bool(self.orchestrator_url.strip())
        )

    def litellm_gateway_enabled(self) -> bool:
        """vLLM stacks always execute via LiteLLM (Nemotron picks, LiteLLM dispatches)."""
        return self.inference_backend.strip().lower() == "vllm"

    @field_validator("api_keys", mode="before")
    @classmethod
    def _parse_api_keys(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @classmethod
    def load(cls, path: Path | None = None, **overrides: Any) -> RouterSettings:
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
