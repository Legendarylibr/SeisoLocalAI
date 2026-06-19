"""Shared environment-variable parsing helpers."""

from __future__ import annotations

import os


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_str(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip()
    return raw if raw else default
