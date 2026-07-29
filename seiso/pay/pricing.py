"""Quote compute with visible protocol fee split (fee added on top)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from seiso.pay.flags import protocol_fee_bps

# Default operator list prices (sats) — override via gateway.yaml / env later.
DEFAULT_PRICES_SATS: dict[str, int] = {
    "inference_per_1k_tokens": 10,
    "inference_flat_call": 5,
    "finetune": 10_000,
    "finetune_smoke": 2_500,
    "slime": 15_000,
    "slime_smoke": 3_000,
    "distill_rl": 12_000,
    "distill_rl_smoke": 2_000,
    "rl_quant": 8_000,
    "rl_quant_minimal": 1_500,
    "nemo_rl": 20_000,
    "nemo_rl_smoke": 4_000,
}

JOB_TYPES = frozenset(
    {"finetune", "slime", "distill_rl", "rl_quant", "nemo_rl", "inference"}
)


@dataclass(frozen=True, slots=True)
class FeeSplit:
    compute_sats: int
    protocol_fee_bps: int
    protocol_fee_sats: int
    total_sats: int
    payee_operator_sats: int
    payee_protocol_sats: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def fee_split(compute_sats: int, *, bps: int | None = None) -> FeeSplit:
    if compute_sats < 0:
        raise ValueError("compute_sats must be >= 0")
    fee_bps = protocol_fee_bps() if bps is None else int(bps)
    if fee_bps < 0:
        raise ValueError("protocol_fee_bps must be >= 0")
    fee = (compute_sats * fee_bps + 9_999) // 10_000  # ceil
    return FeeSplit(
        compute_sats=compute_sats,
        protocol_fee_bps=fee_bps,
        protocol_fee_sats=fee,
        total_sats=compute_sats + fee,
        payee_operator_sats=compute_sats,
        payee_protocol_sats=fee,
    )


def quote_compute(compute_sats: int, *, bps: int | None = None) -> FeeSplit:
    return fee_split(compute_sats, bps=bps)


def price_for_job(
    job_type: str,
    *,
    preset: str | None = None,
    prices: dict[str, int] | None = None,
) -> int:
    table = dict(DEFAULT_PRICES_SATS)
    if prices:
        table.update({str(k): int(v) for k, v in prices.items()})
    jt = job_type.strip().lower()
    if jt not in JOB_TYPES - {"inference"}:
        raise ValueError(f"unknown job type: {job_type!r}")
    preset_l = (preset or "").strip().lower()
    if preset_l in {"smoke", "minimal"}:
        key = f"{jt}_smoke" if jt != "rl_quant" else "rl_quant_minimal"
        if jt == "rl_quant" and preset_l == "minimal":
            key = "rl_quant_minimal"
        elif jt == "rl_quant" and preset_l == "smoke":
            key = "rl_quant_minimal"
        return int(table.get(key, table[jt]))
    return int(table[jt])


def quote_job(
    job_type: str,
    *,
    preset: str | None = None,
    prices: dict[str, int] | None = None,
    bps: int | None = None,
) -> dict[str, Any]:
    compute = price_for_job(job_type, preset=preset, prices=prices)
    split = fee_split(compute, bps=bps)
    out = split.as_dict()
    out.update(
        {
            "job_type": job_type.strip().lower(),
            "preset": (preset or "").strip() or None,
        }
    )
    return out


def quote_inference_tokens(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    prices: dict[str, int] | None = None,
    bps: int | None = None,
    flat_call: bool = False,
) -> dict[str, Any]:
    table = dict(DEFAULT_PRICES_SATS)
    if prices:
        table.update({str(k): int(v) for k, v in prices.items()})
    if flat_call:
        compute = int(table["inference_flat_call"])
    else:
        total_tokens = max(0, int(prompt_tokens)) + max(0, int(completion_tokens))
        per_1k = int(table["inference_per_1k_tokens"])
        compute = max(1, (total_tokens * per_1k + 999) // 1000) if total_tokens else 1
    split = fee_split(compute, bps=bps)
    out = split.as_dict()
    out.update(
        {
            "job_type": "inference",
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "flat_call": flat_call,
        }
    )
    return out
