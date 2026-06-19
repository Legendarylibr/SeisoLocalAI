"""Map RL quantization recommendations to Seiso export settings."""

from __future__ import annotations

import re
from typing import Any

from seiso.export.gguf import normalize_gguf_quant

_BIT_TO_SEISO = {
    2: "q2_k",
    3: "q3_k_m",
    4: "q4_k_m",
    5: "q5_k_m",
    6: "q6_k",
    8: "q8_0",
    16: "f16",
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
            return [normalize_gguf_quant(quant)]
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
            return [normalize_gguf_quant(quant_type)]

    return ["q4_k_m"]
