"""Routing reward — latency, domain fit, wake cost, and response quality heuristics."""

from __future__ import annotations

import re

_ERROR_MARKERS = re.compile(
    r"(error|exception|traceback|failed to|cannot |unable to|out of memory|cuda error)",
    re.IGNORECASE,
)


def compute_route_reward(
    latency_ms: float,
    success: bool,
    *,
    response_text: str = "",
    classified_domain: str = "",
    route_domain_hints: tuple[str, ...] = (),
    wake_latency_ms: float = 0.0,
    complexity_score: float = 0.5,
    fallback_used: bool = False,
) -> float:
    """Composite reward for RL bandit updates (higher is better)."""
    if not success:
        return -1.0

    # Latency: 0–1 scale, penalize slow responses (~30s ceiling)
    total_ms = max(0.0, latency_ms) + max(0.0, wake_latency_ms) * 0.5
    latency_score = max(0.0, 1.0 - (total_ms / 30000.0))

    quality = _response_quality_score(
        response_text,
        complexity_score=complexity_score,
    )

    domain_bonus = 0.0
    if classified_domain and route_domain_hints:
        hints = {d.strip().lower() for d in route_domain_hints}
        if classified_domain.strip().lower() in hints:
            domain_bonus = 0.15

    fallback_penalty = -0.1 if fallback_used else 0.0

    # Weighted blend — quality weighted higher than raw speed
    reward = 0.35 * latency_score + 0.40 * quality + domain_bonus + fallback_penalty
    return max(-1.0, min(1.0, reward))


def _response_quality_score(text: str, *, complexity_score: float) -> float:
    stripped = (text or "").strip()
    if not stripped:
        return 0.0

    score = 0.55
    length = len(stripped)

    # Penalize error-like completions
    if _ERROR_MARKERS.search(stripped[:500]):
        score -= 0.35

    # Very short answers on complex prompts are weak
    if complexity_score >= 0.67 and length < 40:
        score -= 0.25
    elif complexity_score >= 0.34 and length < 20:
        score -= 0.15

    # Modest length bonus (caps to avoid rewarding verbosity)
    if length >= 80:
        score += 0.15
    elif length >= 30:
        score += 0.08

    return max(0.0, min(1.0, score))
