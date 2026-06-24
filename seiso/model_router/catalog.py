"""Specialist route catalog for llama.cpp, local vLLM, and cloud LiteLLM backends."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_BACKEND_VLLM = "vllm"
_BACKEND_LLAMACPP = "llamacpp"
_BACKEND_CLOUD_VLLM = "cloud_vllm"
_BACKEND_CLOUD_API = "cloud_api"
_ALLOWED_BACKENDS = {
    _BACKEND_VLLM,
    _BACKEND_LLAMACPP,
    _BACKEND_CLOUD_VLLM,
    _BACKEND_CLOUD_API,
}


@dataclass
class SpecialistRoute:
    """One specialist served locally (llama.cpp / vLLM) or via cloud LiteLLM providers."""

    route_id: str
    llamaswap_model: str
    backend_url: str
    backend_type: str = _BACKEND_VLLM
    domain_hints: tuple[str, ...] = ()
    hardware_hints: tuple[str, ...] = ("any",)
    vram_hot: bool = False
    sleep_level: int = 1
    idle_sleep_sec: int | None = None
    fallback_priority: int = 100
    openai_model_name: str = ""
    orchestrator_alias: str = ""
    litellm_model: str = ""
    api_key_env: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.openai_model_name:
            self.openai_model_name = self.llamaswap_model
        normalized = (self.backend_type or _BACKEND_VLLM).strip().lower()
        if normalized not in _ALLOWED_BACKENDS:
            raise ValueError(f"Unknown backend_type: {self.backend_type!r}")
        self.backend_type = normalized
        if self.backend_type in {_BACKEND_VLLM, _BACKEND_LLAMACPP, _BACKEND_CLOUD_VLLM} and not (
            self.backend_url.strip()
        ):
            raise ValueError(f"route {self.route_id!r} requires backend_url")
        if self.backend_type == _BACKEND_CLOUD_API and not self.litellm_model.strip():
            raise ValueError(f"route {self.route_id!r} requires litellm_model for cloud_api")

    @property
    def is_vllm(self) -> bool:
        """Local vLLM specialist (sleep/wake lifecycle applies)."""
        return self.backend_type == _BACKEND_VLLM

    @property
    def is_llamacpp(self) -> bool:
        return self.backend_type == _BACKEND_LLAMACPP

    @property
    def is_cloud(self) -> bool:
        return self.backend_type in {_BACKEND_CLOUD_VLLM, _BACKEND_CLOUD_API}

    @property
    def uses_litellm(self) -> bool:
        return self.backend_type in {_BACKEND_VLLM, _BACKEND_CLOUD_VLLM, _BACKEND_CLOUD_API}

    @property
    def vllm_url(self) -> str:
        """Backward-compatible alias for direct backend URL."""
        return self.backend_url

    def matches_hardware(self, hardware: str) -> bool:
        hw = hardware.strip().lower()
        hints = {h.strip().lower() for h in self.hardware_hints}
        return "any" in hints or hw in hints

    def effective_idle_sec(self, default: int) -> int:
        if self.vram_hot:
            return 0
        if self.idle_sleep_sec is not None:
            return int(self.idle_sleep_sec)
        return default

    def orchestrator_detail(self) -> str:
        """Human-readable line for Nemotron tool schema (incl. optional pricing)."""
        desc = self.metadata.get("description") or self.route_id
        hints = ", ".join(self.domain_hints) if self.domain_hints else "general"
        parts = [f"{desc} (domains: {hints})"]
        pricing = self.metadata.get("pricing")
        if isinstance(pricing, dict) and pricing:
            cost_in = pricing.get("input_per_million")
            cost_out = pricing.get("output_per_million")
            latency = pricing.get("avg_latency_sec")
            if cost_in is not None and cost_out is not None:
                parts.append(f"cost ${cost_in}/${cost_out} per M in/out tokens")
            if latency is not None:
                parts.append(f"~{latency}s latency")
        elif self.metadata.get("tier"):
            parts.append(f"tier={self.metadata['tier']}")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "llamaswap_model": self.llamaswap_model,
            "backend_url": self.backend_url,
            "backend_type": self.backend_type,
            "domain_hints": list(self.domain_hints),
            "hardware_hints": list(self.hardware_hints),
            "vram_hot": self.vram_hot,
            "sleep_level": self.sleep_level,
            "idle_sleep_sec": self.idle_sleep_sec,
            "fallback_priority": self.fallback_priority,
            "openai_model_name": self.openai_model_name,
            "orchestrator_alias": self.orchestrator_alias,
            "litellm_model": self.litellm_model,
            "api_key_env": self.api_key_env,
            "metadata": dict(self.metadata),
        }


@dataclass
class SpecialistCatalog:
    routes: list[SpecialistRoute] = field(default_factory=list)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for route in self.routes:
            if route.route_id in seen:
                raise ValueError(f"Duplicate route_id: {route.route_id!r}")
            seen.add(route.route_id)

    def __len__(self) -> int:
        return len(self.routes)

    def __iter__(self):
        return iter(self.routes)

    def by_id(self, route_id: str) -> SpecialistRoute:
        for route in self.routes:
            if route.route_id == route_id:
                return route
        raise KeyError(route_id)

    def by_llamaswap_model(self, model_name: str) -> SpecialistRoute | None:
        key = model_name.strip()
        for route in self.routes:
            if route.llamaswap_model == key or route.openai_model_name == key:
                return route
        return None

    def known_domains(self) -> tuple[str, ...]:
        domains: set[str] = set()
        for route in self.routes:
            domains.update(route.domain_hints)
        return tuple(sorted(domains))

    def litellm_routes(self) -> list[SpecialistRoute]:
        return [r for r in self.routes if r.uses_litellm]

    @classmethod
    def from_json(cls, path: Path) -> SpecialistCatalog:
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw.get("routes", raw if isinstance(raw, list) else [])
        routes: list[SpecialistRoute] = []
        for item in items:
            backend_url = str(item.get("backend_url") or item.get("vllm_url") or "")
            routes.append(
                SpecialistRoute(
                    route_id=str(item["route_id"]),
                    llamaswap_model=str(item.get("llamaswap_model", item["route_id"])),
                    backend_url=backend_url.rstrip("/"),
                    backend_type=str(item.get("backend_type", _BACKEND_VLLM)),
                    domain_hints=tuple(item.get("domain_hints", [])),
                    hardware_hints=tuple(item.get("hardware_hints", ["any"])),
                    vram_hot=bool(item.get("vram_hot", False)),
                    sleep_level=int(item.get("sleep_level", 1)),
                    idle_sleep_sec=item.get("idle_sleep_sec"),
                    fallback_priority=int(item.get("fallback_priority", 100)),
                    openai_model_name=str(item.get("openai_model_name", "")),
                    orchestrator_alias=str(item.get("orchestrator_alias", "")),
                    litellm_model=str(item.get("litellm_model", "")),
                    api_key_env=str(item.get("api_key_env", "")),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        return cls(routes=routes)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"routes": [r.to_dict() for r in self.routes]}, indent=2) + "\n",
            encoding="utf-8",
        )
