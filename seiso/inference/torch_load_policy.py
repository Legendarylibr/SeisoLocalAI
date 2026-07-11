"""Resident Torch model precision policy."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from seiso.env import env_int, env_str

_TORCH_LOAD_PRECISIONS = {"auto", "bf16", "fp16", "4bit"}


@dataclass(frozen=True, slots=True)
class TorchLoadPolicy:
    precision: str
    load_in_4bit: bool
    dtype: str | None
    weight_mb: int
    required_mb: int
    headroom_mb: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_local_weight_mb(model_path: str) -> int:
    """Sum local model weight shards without loading tensors."""
    path = Path(model_path).expanduser()
    if path.is_file():
        return max(1, int(path.stat().st_size / (1024**2)))
    if not path.is_dir():
        return 0
    suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    total = sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in suffixes
    )
    return max(1, int(total / (1024**2))) if total else 0


def resolve_torch_load_policy(
    model_path: str,
    *,
    free_mb: int,
    force_4bit: bool | None = None,
) -> TorchLoadPolicy:
    """Prefer native half precision only when weights plus runtime slack fit."""
    requested = env_str("SEISO_TORCH_LOAD_PRECISION", "auto").strip().lower()
    if requested not in _TORCH_LOAD_PRECISIONS:
        requested = "auto"
    if force_4bit is True:
        requested = "4bit"
    elif force_4bit is False and requested == "auto":
        requested = "bf16"

    weight_mb = estimate_local_weight_mb(model_path)
    reserve = max(
        env_int("SEISO_TORCH_LOAD_RESERVE_MB", 2048),
        int(max(0, free_mb) * 0.25),
    )
    required_mb = int(weight_mb * 1.10) + reserve if weight_mb > 0 else 0
    native_half_fits = (
        platform.system() == "Linux"
        and free_mb > 0
        and weight_mb > 0
        and required_mb <= free_mb
    )
    if requested == "4bit" or (requested == "auto" and not native_half_fits):
        reason = (
            "explicit 4-bit precision"
            if requested == "4bit"
            else "native half precision does not fit reserved headroom"
        )
        return TorchLoadPolicy(
            precision="4bit",
            load_in_4bit=True,
            dtype=None,
            weight_mb=weight_mb,
            required_mb=required_mb,
            headroom_mb=max(0, free_mb),
            reason=reason,
        )

    dtype = "float16"
    precision = "fp16"
    if requested != "fp16":
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = "bfloat16"
                precision = "bf16"
        except ImportError:
            pass
    if requested == "bf16" and precision == "fp16":
        reason = "BF16 requested but unsupported; using FP16"
    elif requested != "auto":
        reason = f"explicit {requested} precision"
    else:
        reason = "native half precision fits reserved headroom"
    return TorchLoadPolicy(
        precision=precision,
        load_in_4bit=False,
        dtype=dtype,
        weight_mb=weight_mb,
        required_mb=required_mb,
        headroom_mb=max(0, free_mb),
        reason=reason,
    )
