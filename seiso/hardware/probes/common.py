"""Shared types and helpers for hardware probes."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SERIAL_RE = re.compile(r"\b(serial|s/n|uuid)[:\s#-]*[\w-]+", re.I)
_HOST_RE = re.compile(r"@[\w.-]+")


@dataclass(frozen=True)
class GpuMemoryProcess:
    pid: int
    process_name: str
    used_mb: int


def sanitize_hardware_label(raw: str, *, max_len: int = 64) -> str:
    text = _SERIAL_RE.sub("", raw)
    text = _HOST_RE.sub("", text)
    text = " ".join(text.split())
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text or "Unknown"
