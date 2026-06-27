"""Shared metric coercion helpers for experiment reports."""

from __future__ import annotations

from typing import Any


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def finite_floats(values: list[Any]) -> list[float]:
    return [out for value in values if (out := finite_float(value)) is not None]
