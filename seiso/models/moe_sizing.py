"""Shared Mixture-of-Experts detection and sizing helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_PARAM_RE = re.compile(r"(?<![a-z])(\d+(?:\.\d+)?)\s*b", re.I)
_ACTIVE_RE = re.compile(r"(?:^|[^a-z0-9])a(\d+(?:\.\d+)?)b(?:$|[^a-z0-9])", re.I)
_MOE_RE = re.compile(
    r"(?:mixtral|moe|deepseek[-_ ]?(?:v2|v3|v4)|qwen[-_ ]?\d+(?:\.\d+)?[-_ ]?moe)",
    re.I,
)
_MIXTRAL_SIZE_RE = re.compile(r"mixtral[-_ ]?8x(7|22)b", re.I)
_MIXTRAL_SIZES = {
    "7": (46.7, 12.9),
    "22": (140.6, 39.1),
}


@dataclass(frozen=True, slots=True)
class MoESizing:
    """Total residency and active-per-token sizing for an MoE model."""

    is_moe: bool
    total_params_b: float | None = None
    active_params_b: float | None = None
    experts_total: int | None = None
    experts_per_tok: int | None = None
    load_vram_mb: int | None = None

    @property
    def compute_note(self) -> str | None:
        if not self.is_moe:
            return None
        pieces: list[str] = []
        if self.active_params_b is not None:
            pieces.append(f"~{self.active_params_b:g}B active/token")
        if self.total_params_b is not None:
            pieces.append(f"~{self.total_params_b:g}B resident")
        return " · ".join(pieces) or "MoE model"


def _positive_number(value: Any, *, billions: bool = False) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    if billions and number > 1_000:
        return number / 1e9
    return number


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def total_params_from_name(name: str) -> float | None:
    """Read the total parameter label, preferring the non-active ``XB`` token."""
    mixtral = _MIXTRAL_SIZE_RE.search(str(name))
    if mixtral:
        return _MIXTRAL_SIZES[mixtral.group(1)][0]
    match = _PARAM_RE.search(str(name))
    return float(match.group(1)) if match else None


def active_params_from_name(name: str) -> float | None:
    """Read an ``A3B``-style active parameter label."""
    match = _ACTIVE_RE.search(str(name))
    if match:
        return float(match.group(1))
    mixtral = _MIXTRAL_SIZE_RE.search(str(name))
    return _MIXTRAL_SIZES[mixtral.group(1)][1] if mixtral else None


def is_moe_model(model_id: str = "", model: Any | None = None) -> bool:
    """Detect an MoE model from config metadata or stable name hints."""
    config = getattr(model, "config", model)
    if config is not None:
        for attr in ("num_local_experts", "num_experts", "n_routed_experts"):
            experts = _positive_int(getattr(config, attr, None))
            if experts and experts > 1:
                return True
        model_type = str(getattr(config, "model_type", "") or "")
        architectures = " ".join(
            str(value) for value in (getattr(config, "architectures", None) or ())
        )
        if _MOE_RE.search(f"{model_type} {architectures}"):
            return True
    text = str(model_id)
    return bool(active_params_from_name(text) is not None or _MOE_RE.search(text))


def sizing_from_config(
    config: Any,
    *,
    model_id: str = "",
    size_bytes: int = 0,
    quant_bytes_per_param: float | None = None,
) -> MoESizing:
    """Build best-effort MoE sizing from a Transformers config and name metadata."""
    is_moe = is_moe_model(model_id, config)
    total = None
    for attr in ("num_parameters", "parameter_count", "total_parameter_count"):
        total = _positive_number(getattr(config, attr, None), billions=True)
        if total is not None:
            break
    total = total or total_params_from_name(model_id)

    experts_total = None
    for attr in ("num_local_experts", "num_experts", "n_routed_experts"):
        experts_total = _positive_int(getattr(config, attr, None))
        if experts_total is not None:
            break
    experts_per_tok = None
    for attr in (
        "num_experts_per_tok",
        "num_selected_experts",
        "num_activated_experts",
        "num_experts_per_token",
    ):
        experts_per_tok = _positive_int(getattr(config, attr, None))
        if experts_per_tok is not None:
            break

    active = None
    for attr in ("active_parameter_count", "num_active_parameters"):
        active = _positive_number(getattr(config, attr, None), billions=True)
        if active is not None:
            break
    active = active or active_params_from_name(model_id)
    if (
        active is None
        and total is not None
        and experts_total
        and experts_per_tok
    ):
        active = total / experts_total * experts_per_tok
    if active is None and is_moe and total is not None:
        active = max(total * 0.2, 1.0)

    load_vram_mb = int(size_bytes / 1024**2) + 512 if size_bytes > 0 else None
    if load_vram_mb is None and total is not None and quant_bytes_per_param:
        load_vram_mb = int(total * quant_bytes_per_param * 1024 + 512)
    return MoESizing(
        is_moe=is_moe,
        total_params_b=total,
        active_params_b=active if is_moe else total,
        experts_total=experts_total,
        experts_per_tok=experts_per_tok,
        load_vram_mb=load_vram_mb,
    )


def sizing_from_reference(
    model_id: str,
    *,
    config: Any | None = None,
    size_bytes: int = 0,
    quant_bytes_per_param: float | None = None,
) -> MoESizing:
    """Build sizing when only an id/path and optional config are available."""
    if config is not None:
        return sizing_from_config(
            config,
            model_id=model_id,
            size_bytes=size_bytes,
            quant_bytes_per_param=quant_bytes_per_param,
        )
    is_moe = is_moe_model(model_id)
    total = total_params_from_name(model_id)
    active = active_params_from_name(model_id)
    if active is None:
        active = max(total * 0.2, 1.0) if is_moe and total is not None else total
    load_vram_mb = int(size_bytes / 1024**2) + 512 if size_bytes > 0 else None
    if load_vram_mb is None and total is not None and quant_bytes_per_param:
        load_vram_mb = int(total * quant_bytes_per_param * 1024 + 512)
    return MoESizing(
        is_moe=is_moe,
        total_params_b=total,
        active_params_b=active,
        load_vram_mb=load_vram_mb,
    )
