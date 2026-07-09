from __future__ import annotations

import math
import statistics

import pytest

from seiso.adaptive_quant import math_utils
from seiso.adaptive_quant.backends.simulator import SimulatorBackend
from seiso.adaptive_quant.types import HardwareType, QuantMode


def test_native_math_availability_probe_returns_bool():
    assert isinstance(math_utils.native_math_available(), bool)


def test_core_math_helpers_match_python_semantics():
    values = [1.5, -2.0, 4.25, 8.0]
    other = [0.5, 3.0, -1.0, 2.0]

    assert math_utils.mean([]) == 0.0
    assert math_utils.mean(values) == pytest.approx(sum(values) / len(values))
    assert math_utils.variance([42.0]) == 0.0
    assert math_utils.variance(values) == pytest.approx(
        sum((value - (sum(values) / len(values))) ** 2 for value in values)
        / len(values)
    )
    assert math_utils.mean_variance(values) == pytest.approx(
        (
            sum(values) / len(values),
            sum((value - (sum(values) / len(values))) ** 2 for value in values)
            / len(values),
        )
    )
    assert math_utils.dot(values, other) == pytest.approx(
        sum(lhs * rhs for lhs, rhs in zip(values, other, strict=True))
    )
    assert math_utils.norm(values) == pytest.approx(
        math.sqrt(sum(value * value for value in values))
    )
    assert math_utils.argmax([1.0, 3.0, 3.0, 2.0]) == 1


def test_softmax_is_stable_and_normalized():
    result = math_utils.softmax([1000.0, 1001.0, 999.0])
    assert sum(result) == pytest.approx(1.0)
    assert result[1] > result[0] > result[2]
    assert math_utils.softmax([]) == []


def test_sigmoid_handles_extreme_values():
    assert math_utils.stable_sigmoid(1000.0) == pytest.approx(1.0)
    assert math_utils.stable_sigmoid(-1000.0) == pytest.approx(0.0)
    assert math_utils.stable_sigmoid(0.0) == pytest.approx(0.5)


def test_ratio_and_std_helpers():
    assert math_utils.safe_ratio(10.0, 2.0) == pytest.approx(5.0)
    assert math_utils.safe_ratio(10.0, 0.0) is None
    assert math_utils.ratio_mean(
        [10.0, 12.0, math.inf], [5.0, 3.0, 2.0]
    ) == pytest.approx(3.0)
    assert math_utils.ratio_mean([1000.0], [1.0], clamp=(0.01, 100.0)) == 1.0
    assert math_utils.sample_std([1.0, 2.0, 5.0]) == pytest.approx(
        statistics.stdev([1.0, 2.0, 5.0])
    )


def test_layer_bit_transform_helpers_match_formulas():
    layer_stats = [0.2, 0.55, 0.9, 1.2]
    dynamic = math_utils.dynamic_layer_bits(
        4,
        layer_stats,
        complexity=0.7,
        min_bits=2,
        max_bits=8,
    )
    assert dynamic == pytest.approx(
        [
            max(2, min(8, 4 + 2.2 * (0.7 - 0.45) + 1.7 * (layer_stat - 0.55)))
            for layer_stat in layer_stats
        ]
    )

    precision_level = 0.6
    precision_bounds = (0.1, 0.9)
    precision_need = 0.72
    scale_factor = 1.1
    clipping_range = 0.85
    min_bits = 2
    max_bits = 8
    base_bits = min_bits + precision_level * ((max_bits - min_bits) * 0.75)
    midpoint = len(layer_stats) // 2
    learned = math_utils.learned_layer_bits(
        layer_stats,
        precision_level=precision_level,
        precision_bounds=precision_bounds,
        precision_need=precision_need,
        scale_factor=scale_factor,
        clipping_range=clipping_range,
        min_bits=min_bits,
        max_bits=max_bits,
    )
    assert learned == pytest.approx(
        [
            max(
                min_bits,
                min(
                    max_bits,
                    base_bits
                    + 1.05 * (layer_stat - 0.55)
                    + 0.80 * (precision_need - 0.50)
                    + (scale_factor - 1.0) * 0.45
                    + (clipping_range - 1.0) * 0.35
                    + (0.12 if layer_index >= midpoint else -0.04),
                ),
            )
            for layer_index, layer_stat in enumerate(layer_stats)
        ]
    )


def test_moe_summary_helpers_match_formulas():
    indices = [0, 2, 1, 2]
    resident = [1.0, 0.0, 0.25, 0.0]
    probabilities = [0.4, 0.3, 0.2, 0.1]
    hotness = [0.9, 0.5, 0.2, 0.8]
    default_index = 1
    variant_count = 3
    denom = max(1, variant_count - 1)

    aggressiveness, churn = math_utils.moe_variant_summary(
        indices,
        default_index=default_index,
        variant_count=variant_count,
    )
    assert aggressiveness == pytest.approx(
        sum(index / denom for index in indices) / len(indices)
    )
    assert churn == pytest.approx(
        sum(abs(index - default_index) / denom for index in indices) / len(indices)
    )

    assert math_utils.moe_swap_cost(
        indices,
        resident,
        probabilities,
        hotness,
        variant_count=variant_count,
    ) == pytest.approx(
        sum(
            (1.2 + 3.4 * (index / denom)) * (0.75 + probability) * (1.10 - 0.35 * hot)
            for index, res, probability, hot in zip(
                indices, resident, probabilities, hotness, strict=True
            )
            if res < 0.5
        )
    )


def test_dot_length_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        math_utils.dot([1.0], [1.0, 2.0])


@pytest.mark.skipif(
    not math_utils.native_math_available(),
    reason="pybind11 extension is not built in this environment",
)
def test_native_extension_exports_expected_hotpath_functions():
    native = math_utils._math_ext
    assert native.mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert native.variance([1.0, 2.0, 3.0]) == pytest.approx(2.0 / 3.0)
    assert tuple(native.mean_variance([1.0, 2.0, 3.0])) == pytest.approx(
        (2.0, 2.0 / 3.0)
    )
    assert native.softmax([0.0, 1.0])[1] > native.softmax([0.0, 1.0])[0]
    assert hasattr(native, "simulator_core_metrics")
    assert hasattr(native, "weighted_reward")
    assert hasattr(native, "extract_input_features")
    assert hasattr(native, "estimate_layer_sensitivity")
    assert hasattr(native, "expand_group_bits")
    assert hasattr(native, "finalize_effective_layer_bits")
    assert hasattr(native, "matrix_vector_add_flat")
    assert hasattr(native, "FlatMatrixHead")
    assert hasattr(native, "FlatValueHead")


def test_finalize_helpers_match_python_semantics():
    assert math_utils.expand_group_bits([2, 4], 6) == pytest.approx(
        [2.0, 2.0, 2.0, 4.0, 4.0, 4.0]
    )
    assert math_utils.pad_or_truncate([1, 2], 4, fill=9) == [1, 2, 9, 9]
    assert math_utils.pad_or_truncate([1, 2, 3, 4], 2, fill=0) == [1, 2]
    assert math_utils.nearest_allowed_bit_width(5, [2, 4, 8], default=4) == 4
    assert math_utils.nearest_allowed_bit_width(None, [2, 4, 8], default=4) == 4


def test_feature_extraction_helpers_are_stable():
    from seiso.adaptive_quant.features import (
        estimate_layer_sensitivity,
        extract_input_features,
        summarize_precision_needs,
        tokenize,
    )
    from seiso.adaptive_quant.types import PromptSample

    assert tokenize("Hello, World_1!") == ["hello", ",", "world_1", "!"]
    prompt = PromptSample(
        prompt_id="p1",
        domain="code",
        text="def add(a, b): return a + b",
    )
    features = extract_input_features(prompt)
    assert features.prompt_length > 0
    assert 0.0 <= features.token_entropy <= 1.0
    sensitivity = estimate_layer_sensitivity(prompt, features, num_layers=4)
    assert len(sensitivity.layer_stats) == 4
    need = summarize_precision_needs(features, sensitivity)
    assert 0.0 <= need <= 1.4


@pytest.mark.skipif(
    not math_utils.native_math_available(),
    reason="pybind11 extension is not built in this environment",
)
def test_native_feature_extraction_matches_python_fallback(monkeypatch):
    from seiso.adaptive_quant import features
    from seiso.adaptive_quant.types import PromptSample

    prompt = PromptSample(
        prompt_id="native-parity",
        domain="math",
        text="Solve 2*x + 3 = 11 for x, please!",
    )
    native_features = features.extract_input_features(prompt)
    native_sensitivity = features.estimate_layer_sensitivity(
        prompt, native_features, num_layers=6
    )
    native_need = features.summarize_precision_needs(native_features, native_sensitivity)

    monkeypatch.setattr(features, "_math_ext", None)
    monkeypatch.setattr(math_utils, "_math_ext", None)
    python_features = features.extract_input_features(prompt)
    python_sensitivity = features.estimate_layer_sensitivity(
        prompt, python_features, num_layers=6
    )
    python_need = features.summarize_precision_needs(python_features, python_sensitivity)

    assert native_features.prompt_length == python_features.prompt_length
    assert native_features.token_entropy == pytest.approx(python_features.token_entropy)
    assert native_features.token_variance == pytest.approx(python_features.token_variance)
    assert native_features.embedding_norm == pytest.approx(python_features.embedding_norm)
    assert native_features.complexity_score == pytest.approx(
        python_features.complexity_score
    )
    assert native_sensitivity.attention_sensitivity == pytest.approx(
        python_sensitivity.attention_sensitivity
    )
    assert native_sensitivity.ffn_sensitivity == pytest.approx(
        python_sensitivity.ffn_sensitivity
    )
    assert native_sensitivity.layer_stats == pytest.approx(
        python_sensitivity.layer_stats
    )
    assert native_need == pytest.approx(python_need)


@pytest.mark.skipif(
    not math_utils.native_math_available(),
    reason="pybind11 extension is not built in this environment",
)
def test_native_finalize_effective_layer_bits_matches_modes():
    result = math_utils.finalize_effective_layer_bits(
        mode="grouped",
        num_layers=6,
        base_bit_width=4,
        group_bit_widths=[2, 8],
        layer_bit_widths=[],
        allowed=[2, 4, 8],
        default_bits=4,
        layer_stats=[0.5] * 6,
        complexity=0.5,
        precision_level=0.5,
        precision_bounds=(0.0, 1.0),
        precision_need=0.5,
        scale_factor=1.0,
        clipping_range=1.0,
    )
    assert result is not None
    effective, avg, var, _base, group, _layer = result
    assert group == [2, 8]
    assert effective == pytest.approx([2.0, 2.0, 2.0, 8.0, 8.0, 8.0])
    assert avg == pytest.approx(5.0)
    assert var == pytest.approx(9.0)


@pytest.mark.skipif(
    not math_utils.native_math_available(),
    reason="pybind11 extension is not built in this environment",
)
def test_native_flat_heads_match_list_semantics():
    from seiso.adaptive_quant.policy_heads import CategoricalHead, ValueHead

    rng = __import__("random").Random(7)
    head = CategoricalHead(3, 2, rng)
    state = [0.5, -1.0, 2.0]
    logits = head.logits(state)
    assert len(logits) == 2
    probs = math_utils.softmax(logits)
    head.update(state, 1, probs, advantage=0.5, learning_rate=0.01)
    assert len(head.weights) == 2
    assert len(head.bias) == 2

    value = ValueHead(3, rng, zero_init=True)
    assert value.predict(state) == pytest.approx(0.0)
    value.update(state, target=1.0, learning_rate=0.1)
    assert value.predict(state) != pytest.approx(0.0)


@pytest.mark.skipif(
    not math_utils.native_math_available(),
    reason="pybind11 extension is not built in this environment",
)
def test_native_policy_head_hotpaths_match_python_semantics():
    weights = [[0.2, -0.4, 0.1], [0.5, 0.0, -0.3]]
    bias = [0.05, -0.2]
    state = [1.0, 2.0, -1.0]
    probabilities = [0.35, 0.65]

    logits = math_utils.matrix_vector_add(weights, bias, state)
    assert logits == pytest.approx(
        [
            sum(lhs * rhs for lhs, rhs in zip(row, state, strict=True)) + b
            for row, b in zip(weights, bias, strict=True)
        ]
    )

    expected_weights = [list(row) for row in weights]
    expected_bias = list(bias)
    selected_index = 1
    advantage = 0.75
    learning_rate = 0.03
    for row_index, row in enumerate(expected_weights):
        coefficient = (
            (1.0 if row_index == selected_index else 0.0) - probabilities[row_index]
        ) * advantage
        for column_index, value in enumerate(state):
            row[column_index] += learning_rate * coefficient * value
        expected_bias[row_index] += learning_rate * coefficient

    assert math_utils.categorical_update(
        weights,
        bias,
        state,
        selected_index,
        probabilities,
        advantage,
        learning_rate,
    )
    for row, expected_row in zip(weights, expected_weights, strict=True):
        assert row == pytest.approx(expected_row)
    assert bias == pytest.approx(expected_bias)


@pytest.mark.skipif(
    not math_utils.native_math_available(),
    reason="pybind11 extension is not built in this environment",
)
def test_native_gaussian_and_value_updates_match_python_semantics():
    weights = [[0.1, 0.2], [-0.1, 0.4]]
    bias = [0.0, 0.3]
    state = [2.0, -1.0]
    raw_samples = [0.6, -0.2]
    raw_means = [0.1, -0.4]
    advantage = 1.2
    learning_rate = 0.05
    variance = 0.25

    expected_weights = [list(row) for row in weights]
    expected_bias = list(bias)
    for row_index, row in enumerate(expected_weights):
        coefficient = (
            (raw_samples[row_index] - raw_means[row_index]) / variance
        ) * advantage
        for column_index, value in enumerate(state):
            row[column_index] += learning_rate * coefficient * value
        expected_bias[row_index] += learning_rate * coefficient

    assert math_utils.gaussian_update(
        weights,
        bias,
        state,
        raw_samples,
        raw_means,
        advantage,
        learning_rate,
        variance,
    )
    for row, expected_row in zip(weights, expected_weights, strict=True):
        assert row == pytest.approx(expected_row)
    assert bias == pytest.approx(expected_bias)

    value_weights = [0.4, -0.2]
    expected_value = [
        weight + learning_rate * 0.9 * value
        for weight, value in zip(value_weights, state, strict=True)
    ]
    assert math_utils.value_update(value_weights, state, 0.9, learning_rate)
    assert value_weights == pytest.approx(expected_value)


@pytest.mark.skipif(
    math_utils.simulator_core_metrics(
        mode="discrete",
        hardware_type="gpu",
        avg_bits=4.0,
        bit_variance=0.0,
        complexity=0.5,
        sensitivity=0.6,
        prompt_length=32.0,
        latency_bias=1.0,
        compute_factor=1.0,
        throughput_bias=1.0,
        kernel_uniformity_preference=0.5,
        preferred_bits=4.0,
        memory_budget_mb=8192.0,
        scale_factor=1.0,
        clipping_range=1.0,
    )
    is None,
    reason="pybind11 simulator core is not built in this environment",
)
def test_native_simulator_core_matches_python_formula():
    kwargs = {
        "mode": QuantMode.LEARNED,
        "hardware_type": HardwareType.LOW_RESOURCE,
        "avg_bits": 3.5,
        "bit_variance": 0.7,
        "complexity": 0.82,
        "sensitivity": 0.64,
        "prompt_length": 96.0,
        "latency_bias": 1.35,
        "compute_factor": 0.52,
        "throughput_bias": 0.72,
        "kernel_uniformity_preference": 0.45,
        "preferred_bits": 4.0,
        "memory_budget_mb": 1024.0,
        "scale_factor": 0.9,
        "clipping_range": 0.95,
    }
    native = math_utils.simulator_core_metrics(
        mode=kwargs["mode"].value,
        hardware_type=kwargs["hardware_type"].value,
        avg_bits=kwargs["avg_bits"],
        bit_variance=kwargs["bit_variance"],
        complexity=kwargs["complexity"],
        sensitivity=kwargs["sensitivity"],
        prompt_length=kwargs["prompt_length"],
        latency_bias=kwargs["latency_bias"],
        compute_factor=kwargs["compute_factor"],
        throughput_bias=kwargs["throughput_bias"],
        kernel_uniformity_preference=kwargs["kernel_uniformity_preference"],
        preferred_bits=kwargs["preferred_bits"],
        memory_budget_mb=kwargs["memory_budget_mb"],
        scale_factor=kwargs["scale_factor"],
        clipping_range=kwargs["clipping_range"],
    )
    python = SimulatorBackend._evaluate_python_core(**kwargs)
    assert native == pytest.approx(python)


@pytest.mark.skipif(
    math_utils.weighted_reward(
        alpha_latency=0.0,
        beta_throughput=0.0,
        gamma_perplexity=0.0,
        delta_memory=0.0,
        epsilon_instability=0.0,
        eta_token_latency=0.0,
        zeta_perplexity_over_ref=0.0,
        theta_kernel_speedup=0.0,
        iota_kernel_latency=0.0,
        latency_ms=1.0,
        throughput_tps=1.0,
        perplexity=1.0,
        memory_mb=1.0,
        latency_ms_per_token=0.0,
        stability_penalty=0.0,
        include_instability=True,
        perplexity_reference=None,
        kernel_speedup=0.0,
        kernel_latency_ms=0.0,
    )
    is None,
    reason="pybind11 reward core is not built in this environment",
)
def test_native_weighted_reward_matches_python_formula():
    reward = math_utils.weighted_reward(
        alpha_latency=0.2,
        beta_throughput=0.05,
        gamma_perplexity=0.7,
        delta_memory=0.01,
        epsilon_instability=0.3,
        eta_token_latency=0.4,
        zeta_perplexity_over_ref=0.9,
        theta_kernel_speedup=1.1,
        iota_kernel_latency=0.6,
        latency_ms=12.0,
        throughput_tps=44.0,
        perplexity=8.0,
        memory_mb=2048.0,
        latency_ms_per_token=0.25,
        stability_penalty=1.5,
        include_instability=True,
        perplexity_reference=6.5,
        kernel_speedup=1.2,
        kernel_latency_ms=0.4,
    )
    expected = (
        -0.2 * 12.0
        + 0.05 * 44.0
        - 0.7 * 8.0
        - 0.01 * 2048.0
        - 0.4 * 0.25
        - 0.3 * 1.5
        - 0.9 * (8.0 - 6.5)
        + 1.1 * 1.2
        - 0.6 * 0.4
    )
    assert reward == pytest.approx(expected)
