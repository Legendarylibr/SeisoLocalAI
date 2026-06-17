"""Map RL quantization recommendations to Seiso export settings."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_BIT_TO_SEISO = {
    2: "q2_k",
    3: "q3_k_m",
    4: "q4_k_m",
    5: "q5_k_m",
    6: "q6_k",
    8: "q8_0",
    16: "f16",
}

_LABEL_TO_SEISO = {
    "Q2_K": "q2_k",
    "Q3_K_M": "q3_k_m",
    "Q3_K_S": "q3_k_s",
    "Q4_K_M": "q4_k_m",
    "Q4_K_S": "q4_k_s",
    "Q5_K_M": "q5_k_m",
    "Q5_K_S": "q5_k_s",
    "Q6_K": "q6_k",
    "Q8_0": "q8_0",
    "F16": "f16",
}


def recommendation_to_gguf_quants(recommendation: dict[str, Any]) -> list[str]:
    """Extract Seiso gguf_quantizations list from an RL recommendation payload."""
    if not recommendation:
        return ["q4_k_m"]

    decision = recommendation.get("decision")
    if isinstance(decision, dict):
        deploy = decision.get("deploy")
        quant = decision.get("quant_type") or decision.get("gguf_quant")
        if isinstance(quant, str) and quant:
            return [_normalize_quant_label(quant)]
        if deploy == "adaptive_policy":
            return ["q4_k_m", "q8_0"]

    fixed = recommendation.get("recommended_quant")
    if isinstance(fixed, dict):
        signature = str(fixed.get("signature", ""))
        match = re.search(r"base=(\d+)", signature)
        if match:
            bits = int(match.group(1))
            label = _BIT_TO_SEISO.get(bits)
            if label:
                return [label]
        quant_type = fixed.get("quant_type")
        if isinstance(quant_type, str):
            return [_normalize_quant_label(quant_type)]

    return ["q4_k_m"]


def _normalize_quant_label(label: str) -> str:
    upper = label.strip().upper().replace("-", "_")
    if upper in _LABEL_TO_SEISO:
        return _LABEL_TO_SEISO[upper]
    return label.strip().lower()


def load_recommendation_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
