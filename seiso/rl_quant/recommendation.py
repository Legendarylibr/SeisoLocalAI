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


def recommendation_evidence(recommendation: dict[str, Any]) -> dict[str, Any]:
    """Evidence / claim-boundary labels for an RL-quant recommendation payload."""
    evidence = recommendation.get("evidence_level")
    if not isinstance(evidence, str) or not evidence:
        research = recommendation.get("research")
        if isinstance(research, dict):
            evidence = research.get("evidence_level")
    if not isinstance(evidence, str) or not evidence:
        evidence = "unknown"
    claimable = recommendation.get("deploy_quality_claimable")
    if claimable is None:
        claimable = evidence not in {"simulator", "unknown"}
    # Simulator/unknown must never be claimable even if a payload sets the flag.
    if evidence in {"simulator", "unknown"}:
        claimable = False
    note = recommendation.get("deploy_quality_note")
    if not note and evidence == "simulator":
        note = (
            "Simulator evidence only — not deploy-grounded without llama_cpp / "
            "external quality sidecar."
        )
    return {
        "evidence_level": evidence,
        "deploy_quality_claimable": bool(claimable),
        "deploy_quality_note": note,
    }


def recommendation_to_gguf_quants(recommendation: dict[str, Any]) -> list[str]:
    """Extract Seiso gguf_quantizations list from an RL recommendation payload.

    Quant labels are always returned for export wiring; callers must consult
    ``recommendation_evidence`` before treating them as deploy quality.
    """
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
