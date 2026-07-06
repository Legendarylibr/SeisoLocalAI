"""GGUF quantization label parsing and effective-bit heuristics."""

from __future__ import annotations

import re
from pathlib import Path

# Exact effective bits per weight from llama.cpp k-quant notes (preferred when known).
KNOWN_QUANT_BITS: dict[str, float] = {
    "Q2_K": 2.625,
    "IQ2_XXS": 2.0625,
    "IQ2_XS": 2.31,
    "IQ2_S": 2.50,
    "IQ2_M": 2.70,
    "Q3_K_S": 3.50,
    "Q3_K_M": 3.91,
    "Q3_K_L": 4.27,
    "IQ3_XXS": 3.06,
    "IQ3_S": 3.44,
    "IQ3_M": 3.66,
    "Q4_0": 4.55,
    "Q4_1": 4.75,
    "Q4_K_S": 4.58,
    "Q4_K_M": 4.83,
    "Q4_K_XL": 4.83,
    "IQ4_XS": 4.25,
    "IQ4_NL": 4.50,
    "Q5_0": 5.54,
    "Q5_1": 5.74,
    "Q5_K_S": 5.54,
    "Q5_K_M": 5.69,
    "Q6_K": 6.56,
    "Q8_0": 8.50,
    "F16": 16.0,
    "BF16": 16.0,
    "F32": 32.0,
}

_GGUF_QUANT_RE = re.compile(
    r"(?:^|[-_.])("
    r"IQ\d+(?:_[A-Z0-9]+)*"
    r"|Q\d+(?:_[A-Z0-9]+)*"
    r"|F16|BF16|F32"
    r")",
    re.I,
)


def normalize_quant_label(label: str) -> str:
    return label.strip().upper().replace("-", "_").replace(".", "_")


def extract_quant_label_from_text(text: str) -> str | None:
    """Extract a GGUF quant token from a filename or repo slug."""
    if not text:
        return None
    matches = [normalize_quant_label(match) for match in _GGUF_QUANT_RE.findall(text)]
    if not matches:
        return None
    return max(matches, key=len)


def extract_quant_label(
    *,
    name: str = "",
    path: str = "",
    metadata: dict | None = None,
) -> str | None:
    meta = metadata or {}
    for candidate in (
        meta.get("gguf_file"),
        (
            meta.get("gguf_files", [None])[0]
            if isinstance(meta.get("gguf_files"), list)
            else None
        ),
        Path(path).name if path else None,
        name,
    ):
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        label = extract_quant_label_from_text(candidate)
        if label:
            return label
    quant = meta.get("quant")
    if isinstance(quant, str) and quant.strip():
        return normalize_quant_label(quant)
    return None


def effective_bits_for_quant(label: str) -> float:
    """Return effective bits/weight for any GGUF quant label."""
    normalized = normalize_quant_label(label)
    if normalized in KNOWN_QUANT_BITS:
        return KNOWN_QUANT_BITS[normalized]

    upper = normalized
    if upper in {"F32"}:
        return 32.0
    if upper in {"F16", "BF16"}:
        return 16.0
    if "Q8" in upper:
        return 8.5
    if "Q7" in upper:
        return 7.5
    if "Q6" in upper:
        return 6.56
    if "Q5" in upper:
        return 5.7
    if "Q4" in upper or "IQ4" in upper:
        return 4.8
    if "Q3" in upper or "IQ3" in upper:
        return 3.9
    if "Q2" in upper or "IQ2" in upper:
        return 2.6
    if "Q1" in upper or "IQ1" in upper:
        return 1.6

    q_match = re.search(r"Q(\d+)", upper)
    if q_match:
        return max(1.0, min(16.0, float(q_match.group(1))))
    iq_match = re.search(r"IQ(\d+)", upper)
    if iq_match:
        return max(1.0, min(8.0, float(iq_match.group(1)) * 0.9))
    return 4.0


def rank_gguf_filenames(
    filenames: list[str],
    *,
    preferred: str | None = None,
) -> list[str]:
    """Rank GGUF filenames by preferred quant without excluding other variants."""
    preferred_u = normalize_quant_label(preferred) if preferred else None

    def score(name: str) -> tuple[int, int, str]:
        upper = name.upper()
        label = extract_quant_label_from_text(name) or ""
        if preferred_u and preferred_u in upper.replace("-", "_"):
            return (0, -len(name), name)
        if preferred_u and label == preferred_u:
            return (1, -len(name), name)
        known = 2 if label in KNOWN_QUANT_BITS else 3
        return (known, -len(name), name)

    return sorted(filenames, key=score)


# Backward-compatible alias used across adaptive_quant.
QUANT_BITS = KNOWN_QUANT_BITS
