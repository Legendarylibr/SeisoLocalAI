"""Model + quantization **route** catalog.

A *route* is a tuple of ``(model repository, quantized file, target hardware hints)``. Each
route names something a user could fetch with ``huggingface-cli`` (or ``hf``) and feed into a
``llama.cpp``-compatible runner. The bandit learner in :mod:`seiso.adaptive_quant.route_policy`
treats each route as an arm and learns which arm wins for which task / hardware bucket.

The catalog is persisted as a JSON document so multiple runs can share / extend it. Loading
is strict: unknown fields raise so typos in registry contributions fail fast.

Quantization labels are mapped to **effective bits per weight** via :data:`QUANT_BITS`. The
estimates here come from llama.cpp's published k-quant breakdowns (and the F16/F32 baselines)
and are deliberately conservative — the bandit only needs *relative* differences to learn a
ranking, so small absolute errors do not bias the learned ordering.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from seiso.adaptive_quant.configuration.validation import (
    validate_hf_filename,
    validate_hf_model_id,
    validate_hf_revision,
    validate_optional_filesystem_path,
    validate_safe_identifier,
)
from seiso.adaptive_quant.logging_utils import read_json, write_json
from seiso.models.gguf_quant import (
    QUANT_BITS,
    effective_bits_for_quant,
    normalize_quant_label,
)

# Acceptable hardware affinity labels — used as soft hints for the bandit.
_HARDWARE_HINTS: frozenset[str] = frozenset({"gpu", "cpu", "low_resource", "any"})


@dataclass(frozen=True)
class QuantSpec:
    """Description of a quantization label.

    ``effective_bits`` is the *bits per weight* used by reward shaping; it is independent of
    the underlying GGUF metadata. ``family`` is currently always ``"gguf"`` but is plumbed
    through so AWQ / GPTQ / EXL2 routes can be added without breaking the schema.
    """

    label: str
    effective_bits: float
    family: str = "gguf"

    @classmethod
    def from_label(cls, label: str, *, family: str = "gguf") -> QuantSpec:
        normalized = normalize_quant_label(label)
        return cls(
            label=normalized,
            effective_bits=effective_bits_for_quant(normalized),
            family=family,
        )


@dataclass
class ModelRoute:
    """A single route the bandit can choose from."""

    route_id: str
    repo_id: str
    quant_label: str
    filename: str | None = None
    revision: str | None = None
    family: str = "gguf"
    parameters_b: float | None = None
    size_mb: float | None = None
    effective_bits: float | None = None
    hardware_hints: tuple[str, ...] = ("any",)
    domain_hints: tuple[str, ...] = ()
    notes: str = ""
    local_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_safe_identifier("route_id", self.route_id)
        if self.local_path is not None:
            validate_optional_filesystem_path("local_path", self.local_path)
        validate_hf_model_id("repo_id", self.repo_id, require_hub_namespace=True)
        if self.filename is not None:
            validate_hf_filename("filename", self.filename)
            if (
                self.family.strip().lower() == "gguf"
                and not self.filename.lower().endswith(".gguf")
            ):
                raise ValueError(
                    f"GGUF route filename must end with '.gguf', got {self.filename!r}"
                )
        if self.revision is not None:
            validate_hf_revision("revision", self.revision)
        normalized_quant = (
            self.quant_label.strip().upper()
            if isinstance(self.quant_label, str)
            else ""
        )
        if not normalized_quant:
            raise ValueError("quant_label is required for ModelRoute")
        object.__setattr__(self, "quant_label", normalized_quant)

        if self.effective_bits is None:
            spec = QuantSpec.from_label(normalized_quant, family=self.family)
            object.__setattr__(self, "effective_bits", float(spec.effective_bits))
        else:
            object.__setattr__(self, "effective_bits", float(self.effective_bits))

        normalized_hints = tuple(
            hint.strip().lower() for hint in self.hardware_hints if hint
        )
        if not normalized_hints:
            normalized_hints = ("any",)
        unknown = [hint for hint in normalized_hints if hint not in _HARDWARE_HINTS]
        if unknown:
            raise ValueError(
                f"Unknown hardware_hints {unknown!r}; allowed: {sorted(_HARDWARE_HINTS)}"
            )
        object.__setattr__(self, "hardware_hints", normalized_hints)

        normalized_domains = tuple(
            domain.strip().lower() for domain in self.domain_hints if domain
        )
        object.__setattr__(self, "domain_hints", normalized_domains)

        if self.parameters_b is not None and float(self.parameters_b) <= 0:
            raise ValueError("parameters_b must be > 0 when set")
        if self.size_mb is not None and float(self.size_mb) <= 0:
            raise ValueError("size_mb must be > 0 when set")

    def quant_spec(self) -> QuantSpec:
        return QuantSpec(
            label=self.quant_label,
            effective_bits=self.effective_bits or 0.0,
            family=self.family,
        )

    def matches_hardware(self, hardware_value: str) -> bool:
        return "any" in self.hardware_hints or hardware_value in self.hardware_hints

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hardware_hints"] = list(self.hardware_hints)
        payload["domain_hints"] = list(self.domain_hints)
        return payload


@dataclass
class RouteCatalog:
    """JSON-backed registry of :class:`ModelRoute` instances."""

    routes: list[ModelRoute] = field(default_factory=list)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for route in self.routes:
            if route.route_id in seen:
                raise ValueError(f"Duplicate route_id in catalog: {route.route_id!r}")
            seen.add(route.route_id)

    def __len__(self) -> int:
        return len(self.routes)

    def __iter__(self):
        return iter(self.routes)

    def add(self, route: ModelRoute, *, replace_existing: bool = False) -> None:
        for index, existing in enumerate(self.routes):
            if existing.route_id == route.route_id:
                if not replace_existing:
                    raise ValueError(
                        f"Route already registered: {route.route_id!r} (use replace_existing=True to overwrite)"
                    )
                self.routes[index] = route
                return
        self.routes.append(route)

    def remove(self, route_id: str) -> bool:
        for index, existing in enumerate(self.routes):
            if existing.route_id == route_id:
                self.routes.pop(index)
                return True
        return False

    def by_id(self, route_id: str) -> ModelRoute:
        for route in self.routes:
            if route.route_id == route_id:
                return route
        raise KeyError(f"Unknown route_id: {route_id!r}")

    def filter(
        self,
        *,
        hardware: str | None = None,
        domain: str | None = None,
        max_effective_bits: float | None = None,
        max_size_mb: float | None = None,
    ) -> list[ModelRoute]:
        result: list[ModelRoute] = []
        for route in self.routes:
            if hardware is not None and not route.matches_hardware(hardware):
                continue
            if (
                domain is not None
                and route.domain_hints
                and domain.lower() not in route.domain_hints
            ):
                continue
            if (
                max_effective_bits is not None
                and (route.effective_bits or 0.0) > max_effective_bits
            ):
                continue
            if (
                max_size_mb is not None
                and route.size_mb is not None
                and route.size_mb > max_size_mb
            ):
                continue
            result.append(route)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RouteCatalog:
        if not isinstance(data, Mapping):
            raise TypeError(
                f"RouteCatalog payload must be a mapping, got {type(data).__name__}"
            )
        raw_routes = data.get("routes", [])
        if not isinstance(raw_routes, list):
            raise TypeError("RouteCatalog 'routes' must be a list")
        valid_keys = {f.name for f in fields(ModelRoute)}
        routes: list[ModelRoute] = []
        for index, payload in enumerate(raw_routes):
            if not isinstance(payload, Mapping):
                raise TypeError(f"routes[{index}] must be an object")
            unknown = set(payload) - valid_keys
            if unknown:
                raise ValueError(
                    f"Unknown ModelRoute keys at routes[{index}]: {sorted(unknown)}"
                )
            kwargs = dict(payload)
            for key in ("hardware_hints", "domain_hints"):
                if key in kwargs and isinstance(kwargs[key], list):
                    kwargs[key] = tuple(str(value) for value in kwargs[key])
            routes.append(ModelRoute(**kwargs))
        return cls(routes=routes)

    def to_dict(self) -> dict[str, Any]:
        return {"routes": [route.to_dict() for route in self.routes]}

    @classmethod
    def from_file(cls, path: str | Path) -> RouteCatalog:
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Route catalog not found: {target}")
        return cls.from_dict(read_json(target, label="Route catalog"))

    def save(self, path: str | Path) -> str:
        target = Path(path)
        write_json(str(target), self.to_dict())
        return str(target)

    def update_local_path(self, route_id: str, local_path: str | None) -> ModelRoute:
        validate_optional_filesystem_path("local_path", local_path)
        for index, existing in enumerate(self.routes):
            if existing.route_id == route_id:
                updated = replace(existing, local_path=local_path)
                self.routes[index] = updated
                return updated
        raise KeyError(f"Unknown route_id: {route_id!r}")


def default_route_catalog(*, token: str | None = None) -> RouteCatalog:
    """Build a starter route catalog from live Hub search.

    Returns an empty catalog when Hub search is unavailable. Users extend routes via
    ``adaptive-rl-quant-route register`` (see :mod:`run_route_learning`).
    """
    try:
        from seiso.models.catalog import search_catalog

        result = search_catalog(limit=16, token=token)
    except Exception:
        return RouteCatalog(routes=[])

    routes: list[ModelRoute] = []
    seen_repos: set[str] = set()
    for row in result.models:
        repo_id = row.get("repo_id")
        if not isinstance(repo_id, str) or not row.get("gguf_repo"):
            continue
        if repo_id in seen_repos:
            continue
        seen_repos.add(repo_id)

        quant_raw = row.get("quant")
        quant_label = (
            str(quant_raw).strip().upper() if isinstance(quant_raw, str) else "Q4_K_M"
        )
        params_raw = row.get("params")
        parameters_b = None
        if isinstance(params_raw, str) and params_raw.strip().upper() not in {"", "?"}:
            try:
                parameters_b = float(params_raw.strip().upper().rstrip("B"))
            except ValueError:
                parameters_b = None

        slug = repo_id.split("/")[-1].lower()
        slug = "".join(ch if ch.isalnum() else "-" for ch in slug).strip("-")
        route_id = f"{slug}-{quant_label.lower().replace('_', '')}"[:80]

        routes.append(
            ModelRoute(
                route_id=route_id,
                repo_id=repo_id,
                quant_label=quant_label,
                parameters_b=parameters_b,
                hardware_hints=("gpu", "cpu"),
                notes="Discovered from Hugging Face Hub catalog search.",
            )
        )
        if len(routes) >= 8:
            break

    return RouteCatalog(routes=routes)


__all__ = [
    "ModelRoute",
    "QUANT_BITS",
    "QuantSpec",
    "RouteCatalog",
    "default_route_catalog",
]
