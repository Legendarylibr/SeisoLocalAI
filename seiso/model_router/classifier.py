"""Prompt classifier — domain detection + feature extraction for routing."""

from __future__ import annotations

import re
from dataclasses import dataclass

from seiso.rl_quant.bootstrap import ensure_adaptive_quant_importable

ensure_adaptive_quant_importable()

from adaptive_quant.features import extract_input_features  # noqa: E402
from adaptive_quant.route_policy import RouteContext  # noqa: E402
from adaptive_quant.types import PromptSample  # noqa: E402

_CODE_MARKERS = re.compile(
    r"(```|def\s+\w+|function\s+\w+|class\s+\w+|import\s+\w+|#include|=>|console\.log|\.py\b|\.js\b|\.ts\b)",
    re.IGNORECASE,
)
_MATH_MARKERS = re.compile(
    r"(\d+\s*[\+\-\*/\^=]\s*\d+|solve|integral|derivative|equation|calculate|sqrt|algebra)",
    re.IGNORECASE,
)
_QA_MARKERS = re.compile(
    r"^(what|who|when|where|why|how|is|are|can|does)\b",
    re.IGNORECASE,
)
_REASONING_MARKERS = re.compile(
    r"(explain|analyze|reason|compare|evaluate|step by step|think through|pros and cons)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClassificationResult:
    domain: str
    complexity_score: float
    prompt_length: int
    confidence: float
    reasoning: str


def _last_user_text(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                return "\n".join(parts)
    return ""


def detect_domain(text: str, known_domains: tuple[str, ...] = ()) -> tuple[str, float, str]:
    """Rule-based domain classifier with confidence score."""
    text = (text or "").strip()
    if not text:
        return "general", 0.3, "empty prompt"

    scores: dict[str, float] = {
        "code": 0.0,
        "math": 0.0,
        "qa": 0.0,
        "reasoning": 0.0,
        "general": 0.15,
    }

    if _CODE_MARKERS.search(text):
        scores["code"] += 0.55
    if _MATH_MARKERS.search(text):
        scores["math"] += 0.45
    if _QA_MARKERS.search(text.strip()):
        scores["qa"] += 0.35
    if _REASONING_MARKERS.search(text):
        scores["reasoning"] += 0.4

    if len(text) > 400:
        scores["reasoning"] += 0.1

    domain = max(scores, key=scores.get)
    confidence = scores[domain]
    if known_domains:
        allow = {d.strip().lower() for d in known_domains}
        if domain not in allow and "general" in allow:
            domain = "general"
            confidence = max(confidence, 0.25)

    return domain, min(confidence, 1.0), f"heuristic scores={scores}"


def classify_messages(
    messages: list[dict],
    *,
    hardware: str,
    known_domains: tuple[str, ...] = (),
) -> tuple[ClassificationResult, RouteContext]:
    text = _last_user_text(messages)
    domain, domain_conf, domain_reason = detect_domain(text, known_domains)

    sample = PromptSample(prompt_id="router", text=text, domain=domain)
    features = extract_input_features(sample)

    result = ClassificationResult(
        domain=domain,
        complexity_score=features.complexity_score,
        prompt_length=features.prompt_length,
        confidence=domain_conf,
        reasoning=domain_reason,
    )

    context = RouteContext.from_features(
        hardware=hardware,
        domain=domain,
        complexity_score=features.complexity_score,
        known_domains=known_domains,
    )
    return result, context
