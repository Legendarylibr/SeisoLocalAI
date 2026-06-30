from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections.abc import Sequence

try:
    from seiso.adaptive_quant.native import _math_ext
except ImportError:  # pragma: no cover - depends on optional compiled extension
    _math_ext = None


def native_math_available() -> bool:
    return _math_ext is not None


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def stable_sigmoid(value: float) -> float:
    """Numerically stable logistic (sigmoid) for a scalar."""
    if _math_ext is not None:
        return float(_math_ext.stable_sigmoid(value))
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def mean(values: Sequence[float]) -> float:
    if _math_ext is not None:
        return float(_math_ext.mean(values))
    if not values:
        return 0.0
    return sum(values) / len(values)


def variance(values: Sequence[float]) -> float:
    if _math_ext is not None:
        return float(_math_ext.variance(values))
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return sum((value - avg) ** 2 for value in values) / len(values)


def mean_variance(values: Sequence[float]) -> tuple[float, float]:
    """Return population mean and variance in one pass when native math is available."""
    if _math_ext is not None:
        avg, var = _math_ext.mean_variance(values)
        return float(avg), float(var)
    if not values:
        return 0.0, 0.0
    avg = sum(values) / len(values)
    if len(values) < 2:
        return avg, 0.0
    return avg, sum((value - avg) ** 2 for value in values) / len(values)


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    if _math_ext is not None:
        return float(_math_ext.dot(left, right))
    return sum(lhs * rhs for lhs, rhs in zip(left, right, strict=True))


def norm(values: Sequence[float]) -> float:
    if _math_ext is not None:
        return float(_math_ext.norm(values))
    return math.sqrt(sum(value * value for value in values))


def softmax(logits: Sequence[float]) -> list[float]:
    if _math_ext is not None:
        return list(_math_ext.softmax(logits))
    if not logits:
        return []
    max_logit = max(logits)
    shifted = [math.exp(logit - max_logit) for logit in logits]
    total = sum(shifted)
    if total <= 0.0:
        return [1.0 / len(logits)] * len(logits)
    return [value / total for value in shifted]


def matrix_vector_add(
    weights: Sequence[Sequence[float]],
    bias: Sequence[float],
    state_vector: Sequence[float],
) -> list[float]:
    """Compute ``weights @ state_vector + bias`` for policy heads."""
    if _math_ext is not None:
        return list(_math_ext.matrix_vector_add(weights, bias, state_vector))
    return [
        dot(row, state_vector) + bias_value
        for row, bias_value in zip(weights, bias, strict=True)
    ]


def categorical_update(
    weights: list[list[float]],
    bias: list[float],
    state_vector: Sequence[float],
    selected_index: int,
    probabilities: Sequence[float],
    advantage: float,
    learning_rate: float,
) -> bool:
    """Native in-place categorical policy update; returns True when accelerated."""
    if _math_ext is None:
        return False
    _math_ext.categorical_update(
        weights,
        bias,
        state_vector,
        selected_index,
        probabilities,
        advantage,
        learning_rate,
    )
    return True


def gaussian_update(
    weights: list[list[float]],
    bias: list[float],
    state_vector: Sequence[float],
    raw_samples: Sequence[float],
    raw_means: Sequence[float],
    advantage: float,
    learning_rate: float,
    variance: float,
) -> bool:
    """Native in-place Gaussian policy update; returns True when accelerated."""
    if _math_ext is None:
        return False
    _math_ext.gaussian_update(
        weights,
        bias,
        state_vector,
        raw_samples,
        raw_means,
        advantage,
        learning_rate,
        variance,
    )
    return True


def value_update(
    weights: list[float],
    state_vector: Sequence[float],
    error: float,
    learning_rate: float,
) -> bool:
    """Native in-place value-head update; returns True when accelerated."""
    if _math_ext is None:
        return False
    _math_ext.value_update(weights, state_vector, error, learning_rate)
    return True


def simulator_core_metrics(
    *,
    mode: str,
    hardware_type: str,
    avg_bits: float,
    bit_variance: float,
    complexity: float,
    sensitivity: float,
    prompt_length: float,
    latency_bias: float,
    compute_factor: float,
    throughput_bias: float,
    kernel_uniformity_preference: float,
    preferred_bits: float,
    memory_budget_mb: float,
    scale_factor: float,
    clipping_range: float,
) -> tuple[float, float, float, float] | None:
    """Native simulator scoring kernel; returns ``None`` when unavailable."""
    if _math_ext is None or not hasattr(_math_ext, "simulator_core_metrics"):
        return None
    latency, throughput, perplexity, memory = _math_ext.simulator_core_metrics(
        mode,
        hardware_type,
        avg_bits,
        bit_variance,
        complexity,
        sensitivity,
        prompt_length,
        latency_bias,
        compute_factor,
        throughput_bias,
        kernel_uniformity_preference,
        preferred_bits,
        memory_budget_mb,
        scale_factor,
        clipping_range,
    )
    return float(latency), float(throughput), float(perplexity), float(memory)


def weighted_reward(
    *,
    alpha_latency: float,
    beta_throughput: float,
    gamma_perplexity: float,
    delta_memory: float,
    epsilon_instability: float,
    eta_token_latency: float,
    zeta_perplexity_over_ref: float,
    theta_kernel_speedup: float,
    iota_kernel_latency: float,
    latency_ms: float,
    throughput_tps: float,
    perplexity: float,
    memory_mb: float,
    latency_ms_per_token: float,
    stability_penalty: float,
    include_instability: bool,
    perplexity_reference: float | None,
    kernel_speedup: float,
    kernel_latency_ms: float,
) -> float | None:
    """Native reward aggregation; returns ``None`` when unavailable."""
    if _math_ext is None or not hasattr(_math_ext, "weighted_reward"):
        return None
    return float(
        _math_ext.weighted_reward(
            alpha_latency,
            beta_throughput,
            gamma_perplexity,
            delta_memory,
            epsilon_instability,
            eta_token_latency,
            zeta_perplexity_over_ref,
            theta_kernel_speedup,
            iota_kernel_latency,
            latency_ms,
            throughput_tps,
            perplexity,
            memory_mb,
            latency_ms_per_token,
            stability_penalty,
            include_instability,
            perplexity_reference,
            kernel_speedup,
            kernel_latency_ms,
        )
    )


def dynamic_layer_bits(
    base_bit_width: int,
    layer_stats: Sequence[float],
    *,
    complexity: float,
    min_bits: int,
    max_bits: int,
) -> list[float]:
    """Compute per-layer dynamic precision bits for a decision."""
    if _math_ext is not None:
        return list(
            _math_ext.dynamic_layer_bits(
                base_bit_width,
                layer_stats,
                complexity,
                float(min_bits),
                float(max_bits),
            )
        )
    return [
        clamp(
            base_bit_width + 2.2 * (complexity - 0.45) + 1.7 * (layer_stat - 0.55),
            min_bits,
            max_bits,
        )
        for layer_stat in layer_stats
    ]


def learned_layer_bits(
    layer_stats: Sequence[float],
    *,
    precision_level: float,
    precision_bounds: tuple[float, float],
    precision_need: float,
    scale_factor: float,
    clipping_range: float,
    min_bits: int,
    max_bits: int,
) -> list[float]:
    """Compute per-layer learned precision bits for a decision."""
    if _math_ext is not None:
        return list(
            _math_ext.learned_layer_bits(
                layer_stats,
                precision_level,
                precision_bounds[0],
                precision_bounds[1],
                precision_need,
                scale_factor,
                clipping_range,
                float(min_bits),
                float(max_bits),
            )
        )
    learned_span = (max_bits - min_bits) * 0.75
    base_bits = min_bits + clamp(precision_level, *precision_bounds) * learned_span
    midpoint = len(layer_stats) // 2
    return [
        clamp(
            base_bits
            + 1.05 * (layer_stat - 0.55)
            + 0.80 * (precision_need - 0.50)
            + (scale_factor - 1.0) * 0.45
            + (clipping_range - 1.0) * 0.35
            + (0.12 if layer_index >= midpoint else -0.04),
            min_bits,
            max_bits,
        )
        for layer_index, layer_stat in enumerate(layer_stats)
    ]


def moe_variant_summary(
    indices: Sequence[int],
    *,
    default_index: int,
    variant_count: int,
) -> tuple[float, float]:
    """Return average variant aggressiveness and churn for MoE routing."""
    if not indices:
        return 0.0, 0.0
    if _math_ext is not None:
        aggressiveness, churn = _math_ext.moe_variant_summary(
            indices,
            default_index,
            variant_count,
        )
        return float(aggressiveness), float(churn)
    denom = max(1, variant_count - 1)
    average_aggressiveness = sum(index / denom for index in indices) / len(indices)
    variant_churn = sum(abs(index - default_index) / denom for index in indices) / len(
        indices
    )
    return float(average_aggressiveness), float(variant_churn)


def moe_swap_cost(
    indices: Sequence[int],
    resident_on_device: Sequence[float],
    router_probabilities: Sequence[float],
    hotness: Sequence[float],
    *,
    variant_count: int,
) -> float:
    """Predict MoE swap cost for selected variants."""
    if not indices:
        return 0.0
    if _math_ext is not None:
        return float(
            _math_ext.moe_swap_cost(
                indices,
                resident_on_device,
                router_probabilities,
                hotness,
                variant_count,
            )
        )
    denom = max(1, variant_count - 1)
    total = 0.0
    for index, resident, probability, hot in zip(
        indices, resident_on_device, router_probabilities, hotness, strict=True
    ):
        if resident < 0.5:
            aggressiveness = index / denom
            total += (1.2 + 3.4 * aggressiveness) * (0.75 + probability) * (
                1.10 - 0.35 * hot
            )
    return float(total)


def sample_categorical(probabilities: Sequence[float], rng: random.Random) -> int:
    threshold = rng.random()
    running = 0.0
    for index, probability in enumerate(probabilities):
        running += probability
        if threshold <= running:
            return index
    return max(0, len(probabilities) - 1)


def argmax(values: Sequence[float]) -> int:
    if _math_ext is not None:
        return int(_math_ext.argmax(values))
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def discrete_precision_level(
    bit_width: int, discrete_bit_widths: Sequence[int]
) -> float:
    """Map ``bit_width`` to [0, 1] along ``discrete_bit_widths`` range (0.0 when only one width)."""
    lo = min(discrete_bit_widths)
    hi = max(discrete_bit_widths)
    span = hi - lo
    if span <= 0:
        return 0.0
    return float(bit_width - lo) / float(span)


def deterministic_float(key: str, lower: float = 0.0, upper: float = 1.0) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return lower + (upper - lower) * bucket


def stable_hash_int(text: str, modulo: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


def gaussian_sample(mean_value: float, stddev: float, rng: random.Random) -> float:
    if stddev <= 0.0:
        return mean_value
    return rng.gauss(mean_value, stddev)


def safe_ratio(numer: float, denom: float) -> float | None:
    if _math_ext is not None:
        ratio = _math_ext.safe_ratio(numer, denom)
        return None if ratio is None else float(ratio)
    if not math.isfinite(numer) or not math.isfinite(denom) or denom <= 0:
        return None
    return numer / denom


def ratio_mean(
    observed: list[float],
    simulated: list[float],
    *,
    clamp: tuple[float, float] = (0.01, 100.0),
) -> float:
    lower, upper = clamp
    if _math_ext is not None:
        return float(_math_ext.ratio_mean(observed, simulated, lower, upper))
    ratios = [
        r
        for o, s in zip(observed, simulated, strict=False)
        if (r := safe_ratio(o, s)) is not None and lower < r < upper
    ]
    return float(statistics.fmean(ratios)) if ratios else 1.0


def sample_std(values: list[float]) -> float:
    if _math_ext is not None:
        return float(_math_ext.sample_std(values))
    return float(statistics.stdev(values)) if len(values) >= 2 else 0.0


def fmt_float(x: float, *, digits: int = 2) -> str:
    return "nan" if not math.isfinite(x) else f"{x:.{digits}f}"


def parse_seed_list(raw: str) -> list[int]:
    """Parse comma lists (``1,2,3``) or inclusive ranges (``3-5``)."""
    text = raw.strip()
    if not text:
        return []
    if "-" in text and "," not in text:
        left, right = text.split("-", 1)
        start = int(left.strip())
        end = int(right.strip())
        if end < start:
            start, end = end, start
        return list(range(start, end + 1))
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def format_display(value: object, *, style: str = "report", digits: int = 2) -> str:
    """Format numbers for CLI footers, Markdown reports, or finite-only tables."""
    if style == "footer":
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, (int, float)):
        if style == "float" and isinstance(value, float) and not math.isfinite(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric, got bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {result!r}")
    return result


def non_negative_int(value: object, *, label: str, maximum: int | None = None) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, got bool")
    if not isinstance(value, int):
        raise TypeError(f"{label} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{label} must be >= 0, got {value!r}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be <= {maximum}, got {value!r}")
    return int(value)
