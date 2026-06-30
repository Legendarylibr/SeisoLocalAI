from __future__ import annotations

import math
import statistics

import pytest

from seiso.adaptive_quant import math_utils


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
    assert math_utils.ratio_mean([10.0, 12.0, math.inf], [5.0, 3.0, 2.0]) == pytest.approx(
        3.0
    )
    assert math_utils.ratio_mean([1000.0], [1.0], clamp=(0.01, 100.0)) == 1.0
    assert math_utils.sample_std([1.0, 2.0, 5.0]) == pytest.approx(
        statistics.stdev([1.0, 2.0, 5.0])
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
    assert native.softmax([0.0, 1.0])[1] > native.softmax([0.0, 1.0])[0]
