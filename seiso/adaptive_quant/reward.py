from __future__ import annotations

from typing import Protocol

from seiso.adaptive_quant.math_utils import weighted_reward
from seiso.adaptive_quant.types import BackendMetricDict


class _MoEPenaltyConfig(Protocol):
    moe_swap_penalty: float
    moe_cache_miss_penalty: float
    moe_variant_churn_penalty: float


class _RewardWeights(Protocol):
    alpha_latency: float
    beta_throughput: float
    gamma_perplexity: float
    delta_memory: float
    epsilon_instability: float
    eta_token_latency: float
    zeta_perplexity_over_ref: float
    theta_kernel_speedup: float
    iota_kernel_latency: float


def compute_weighted_reward(
    *,
    reward_weights: _RewardWeights,
    metrics: BackendMetricDict,
    stability_penalty: float = 0.0,
    perplexity_reference: float | None = None,
    include_instability: bool = True,
    latency_ms_per_token_default: float = 0.0,
) -> float:
    """Shared reward helper used by multiple pipelines.

    Keeps reward math consistent across the environment trainer and route research while
    letting callers opt out of instability terms when they are not applicable.
    """

    weights = reward_weights
    native_reward = weighted_reward(
        alpha_latency=float(weights.alpha_latency),
        beta_throughput=float(weights.beta_throughput),
        gamma_perplexity=float(weights.gamma_perplexity),
        delta_memory=float(weights.delta_memory),
        epsilon_instability=float(weights.epsilon_instability),
        eta_token_latency=float(weights.eta_token_latency),
        zeta_perplexity_over_ref=float(weights.zeta_perplexity_over_ref),
        theta_kernel_speedup=float(weights.theta_kernel_speedup),
        iota_kernel_latency=float(weights.iota_kernel_latency),
        latency_ms=float(metrics["latency_ms"]),
        throughput_tps=float(metrics["throughput_tps"]),
        perplexity=float(metrics["perplexity"]),
        memory_mb=float(metrics["memory_mb"]),
        latency_ms_per_token=float(
            metrics.get("latency_ms_per_token", latency_ms_per_token_default)
        ),
        stability_penalty=float(stability_penalty),
        include_instability=bool(include_instability),
        perplexity_reference=perplexity_reference,
        kernel_speedup=float(metrics.get("kernel_speedup", 0.0)),
        kernel_latency_ms=float(metrics.get("kernel_latency_ms", 0.0)),
    )
    if native_reward is not None:
        return native_reward

    reward = (
        -weights.alpha_latency * float(metrics["latency_ms"])
        + weights.beta_throughput * float(metrics["throughput_tps"])
        - weights.gamma_perplexity * float(metrics["perplexity"])
        - weights.delta_memory * float(metrics["memory_mb"])
        - weights.eta_token_latency
        * float(metrics.get("latency_ms_per_token", latency_ms_per_token_default))
    )
    if include_instability:
        reward -= weights.epsilon_instability * float(stability_penalty)

    ref = perplexity_reference
    zeta = float(weights.zeta_perplexity_over_ref)
    if ref is not None and zeta > 0.0:
        over = max(0.0, float(metrics["perplexity"]) - float(ref))
        reward -= zeta * over

    kernel_speedup = float(metrics.get("kernel_speedup", 0.0))
    if kernel_speedup > 0.0:
        reward += weights.theta_kernel_speedup * kernel_speedup
    kernel_latency = float(metrics.get("kernel_latency_ms", 0.0))
    if kernel_latency > 0.0:
        reward -= weights.iota_kernel_latency * kernel_latency

    return float(reward)


def apply_moe_reward_penalties(
    reward: float,
    metrics: BackendMetricDict,
    config: _MoEPenaltyConfig,
) -> float:
    """Subtract MoE swap/cache/churn terms shared by env and other reward call sites."""
    adjusted = float(reward)
    adjusted -= config.moe_swap_penalty * float(metrics.get("swap_cost_ms", 0.0))
    adjusted -= config.moe_cache_miss_penalty * float(
        metrics.get("cache_miss_count", 0.0)
    )
    adjusted -= config.moe_variant_churn_penalty * float(
        metrics.get("variant_churn", 0.0)
    )
    return float(adjusted)


__all__ = ["apply_moe_reward_penalties", "compute_weighted_reward"]
