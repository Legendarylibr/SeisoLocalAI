"""Optional external harness adapters (Pi, OMP, Hermes, Cline, OpenClaw)."""

from __future__ import annotations

from seiso.agent.adapters.base import BaseAdapter
from seiso.agent.adapters.cline import ClineAdapter
from seiso.agent.adapters.detect import default_harness_id, detect_all, detect_harness
from seiso.agent.adapters.endpoint import ResolvedEndpoint, resolve_endpoint
from seiso.agent.adapters.hermes import HermesAdapter
from seiso.agent.adapters.openclaw import OpenClawAdapter
from seiso.agent.adapters.pi import OmpAdapter, PiAdapter
from seiso.agent.adapters.types import (
    HARNESS_IDS,
    HARNESS_LABELS,
    DetectedHarness,
    LaunchResult,
    LaunchSpec,
    parse_harness_id,
)

_ADAPTERS = {
    "pi": PiAdapter,
    "omp": OmpAdapter,
    "hermes": HermesAdapter,
    "cline": ClineAdapter,
    "openclaw": OpenClawAdapter,
}


def get_adapter(harness_id: str) -> BaseAdapter:
    hid = parse_harness_id(harness_id)
    return _ADAPTERS[hid]()


__all__ = [
    "HARNESS_IDS",
    "HARNESS_LABELS",
    "ClineAdapter",
    "DetectedHarness",
    "HermesAdapter",
    "LaunchResult",
    "LaunchSpec",
    "OmpAdapter",
    "OpenClawAdapter",
    "PiAdapter",
    "ResolvedEndpoint",
    "default_harness_id",
    "detect_all",
    "detect_harness",
    "get_adapter",
    "parse_harness_id",
    "resolve_endpoint",
]
