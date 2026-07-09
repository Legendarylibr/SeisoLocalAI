"""Policy network heads and checkpoint (de)serialization helpers."""

from __future__ import annotations

import random

from seiso.adaptive_quant.math_utils import (
    _math_ext,
    argmax,
    categorical_update,
    clamp,
    finite_float,
    gaussian_sample,
    gaussian_update,
    matrix_vector_add,
    native_flat_heads_available,
    sample_categorical,
    softmax,
    stable_sigmoid,
    value_update,
)


def _random_matrix(
    rows: int,
    cols: int,
    rng: random.Random,
    scale: float = 0.08,
) -> list[list[float]]:
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


class CategoricalHead:
    def __init__(self, input_dim: int, output_dim: int, rng: random.Random) -> None:
        self._native = None
        if native_flat_heads_available():
            self._native = _math_ext.FlatMatrixHead(output_dim, input_dim)
            weights = _random_matrix(output_dim, input_dim, rng)
            self._native.set_weights(weights)
            self._native.set_bias([0.0] * output_dim)
            self._weights = weights
            self._bias = [0.0] * output_dim
        else:
            self._weights = _random_matrix(output_dim, input_dim, rng)
            self._bias = [0.0] * output_dim

    def __deepcopy__(self, memo: dict[int, object]) -> CategoricalHead:
        # Native pybind storage is not deepcopy-safe; clone via weight/bias props.
        clone = CategoricalHead.__new__(CategoricalHead)
        memo[id(self)] = clone
        rows = len(self.weights)
        cols = len(self.weights[0]) if rows else 0
        clone._native = None
        if native_flat_heads_available():
            clone._native = _math_ext.FlatMatrixHead(rows, cols)
        clone.weights = [list(row) for row in self.weights]
        clone.bias = list(self.bias)
        return clone

    @property
    def weights(self) -> list[list[float]]:
        if self._native is not None:
            self._weights = [list(row) for row in self._native.get_weights()]
        return self._weights

    @weights.setter
    def weights(self, value: list[list[float]]) -> None:
        self._weights = [list(row) for row in value]
        if self._native is not None:
            self._native.set_weights(self._weights)

    @property
    def bias(self) -> list[float]:
        if self._native is not None:
            self._bias = list(self._native.get_bias())
        return self._bias

    @bias.setter
    def bias(self, value: list[float]) -> None:
        self._bias = list(value)
        if self._native is not None:
            self._native.set_bias(self._bias)

    def logits(self, state_vector: list[float]) -> list[float]:
        if self._native is not None:
            return list(self._native.logits(state_vector))
        return matrix_vector_add(self._weights, self._bias, state_vector)

    def sample(
        self,
        state_vector: list[float],
        rng: random.Random,
        deterministic: bool = False,
        *,
        epsilon: float = 0.0,
    ) -> tuple[int, list[float]]:
        probabilities = softmax(self.logits(state_vector))
        if deterministic:
            return argmax(probabilities), probabilities
        if epsilon > 0.0 and rng.random() < float(epsilon):
            return rng.randrange(len(probabilities)), probabilities
        return sample_categorical(probabilities, rng), probabilities

    def update(
        self,
        state_vector: list[float],
        selected_index: int,
        probabilities: list[float],
        advantage: float,
        learning_rate: float,
    ) -> None:
        if self._native is not None:
            self._native.categorical_update(
                state_vector,
                selected_index,
                probabilities,
                advantage,
                learning_rate,
            )
            return
        if categorical_update(
            self._weights,
            self._bias,
            state_vector,
            selected_index,
            probabilities,
            advantage,
            learning_rate,
        ):
            return
        for row_index, row in enumerate(self._weights):
            coefficient = (
                (1.0 if row_index == selected_index else 0.0) - probabilities[row_index]
            ) * advantage
            for column_index, value in enumerate(state_vector):
                row[column_index] += learning_rate * coefficient * value
            self._bias[row_index] += learning_rate * coefficient


class GaussianHead:
    def __init__(
        self, input_dim: int, output_dim: int, rng: random.Random, stddev: float
    ) -> None:
        self.stddev = stddev
        self._native = None
        if native_flat_heads_available():
            self._native = _math_ext.FlatMatrixHead(output_dim, input_dim)
            weights = _random_matrix(output_dim, input_dim, rng)
            self._native.set_weights(weights)
            self._native.set_bias([0.0] * output_dim)
            self._weights = weights
            self._bias = [0.0] * output_dim
        else:
            self._weights = _random_matrix(output_dim, input_dim, rng)
            self._bias = [0.0] * output_dim

    def __deepcopy__(self, memo: dict[int, object]) -> GaussianHead:
        clone = GaussianHead.__new__(GaussianHead)
        memo[id(self)] = clone
        clone.stddev = float(self.stddev)
        rows = len(self.weights)
        cols = len(self.weights[0]) if rows else 0
        clone._native = None
        if native_flat_heads_available():
            clone._native = _math_ext.FlatMatrixHead(rows, cols)
        clone.weights = [list(row) for row in self.weights]
        clone.bias = list(self.bias)
        return clone

    @property
    def weights(self) -> list[list[float]]:
        if self._native is not None:
            self._weights = [list(row) for row in self._native.get_weights()]
        return self._weights

    @weights.setter
    def weights(self, value: list[list[float]]) -> None:
        self._weights = [list(row) for row in value]
        if self._native is not None:
            self._native.set_weights(self._weights)

    @property
    def bias(self) -> list[float]:
        if self._native is not None:
            self._bias = list(self._native.get_bias())
        return self._bias

    @bias.setter
    def bias(self, value: list[float]) -> None:
        self._bias = list(value)
        if self._native is not None:
            self._native.set_bias(self._bias)

    def means(self, state_vector: list[float]) -> list[float]:
        if self._native is not None:
            return list(self._native.logits(state_vector))
        return matrix_vector_add(self._weights, self._bias, state_vector)

    def sample(
        self,
        state_vector: list[float],
        rng: random.Random,
        bounds: list[tuple[float, float]],
        deterministic: bool = False,
    ) -> tuple[list[float], list[float], list[float]]:
        raw_means = self.means(state_vector)
        if deterministic:
            raw_samples = list(raw_means)
        else:
            raw_samples = [
                gaussian_sample(mean_value, self.stddev, rng)
                for mean_value in raw_means
            ]
        mapped = [
            _map_to_bounds(stable_sigmoid(sample), lower, upper)
            for sample, (lower, upper) in zip(raw_samples, bounds, strict=True)
        ]
        return mapped, raw_samples, raw_means

    def update(
        self,
        state_vector: list[float],
        raw_samples: list[float],
        raw_means: list[float],
        advantage: float,
        learning_rate: float,
    ) -> None:
        variance = max(self.stddev * self.stddev, 1e-6)
        if self._native is not None:
            self._native.gaussian_update(
                state_vector,
                raw_samples,
                raw_means,
                advantage,
                learning_rate,
                variance,
            )
            return
        if gaussian_update(
            self._weights,
            self._bias,
            state_vector,
            raw_samples,
            raw_means,
            advantage,
            learning_rate,
            variance,
        ):
            return
        for row_index, row in enumerate(self._weights):
            coefficient = (
                (raw_samples[row_index] - raw_means[row_index]) / variance
            ) * advantage
            for column_index, value in enumerate(state_vector):
                row[column_index] += learning_rate * coefficient * value
            self._bias[row_index] += learning_rate * coefficient


class ValueHead:
    def __init__(
        self, input_dim: int, rng: random.Random, *, zero_init: bool = False
    ) -> None:
        self._native = None
        weights = (
            [0.0 for _ in range(input_dim)]
            if zero_init
            else [rng.uniform(-0.05, 0.05) for _ in range(input_dim)]
        )
        if native_flat_heads_available():
            self._native = _math_ext.FlatValueHead(input_dim)
            self._native.set_weights(weights)
            self._native.set_bias(0.0)
            self._weights = weights
            self._bias = 0.0
        else:
            self._weights = weights
            self._bias = 0.0

    def __deepcopy__(self, memo: dict[int, object]) -> ValueHead:
        clone = ValueHead.__new__(ValueHead)
        memo[id(self)] = clone
        clone._native = None
        if native_flat_heads_available():
            clone._native = _math_ext.FlatValueHead(len(self.weights))
        clone.weights = list(self.weights)
        clone.bias = float(self.bias)
        return clone

    @property
    def weights(self) -> list[float]:
        if self._native is not None:
            self._weights = list(self._native.get_weights())
        return self._weights

    @weights.setter
    def weights(self, value: list[float]) -> None:
        self._weights = list(value)
        if self._native is not None:
            self._native.set_weights(self._weights)

    @property
    def bias(self) -> float:
        if self._native is not None:
            self._bias = float(self._native.get_bias())
        return self._bias

    @bias.setter
    def bias(self, value: float) -> None:
        self._bias = float(value)
        if self._native is not None:
            self._native.set_bias(self._bias)

    def predict(self, state_vector: list[float]) -> float:
        if self._native is not None:
            return float(self._native.predict(state_vector))
        return matrix_vector_add([self._weights], [self._bias], state_vector)[0]

    def update(
        self, state_vector: list[float], target: float, learning_rate: float
    ) -> None:
        if self._native is not None:
            self._native.update(state_vector, target, learning_rate)
            return
        prediction = self.predict(state_vector)
        error = target - prediction
        if value_update(self._weights, state_vector, error, learning_rate):
            self._bias += learning_rate * error
            return
        for index, value in enumerate(state_vector):
            self._weights[index] += learning_rate * error * value
        self._bias += learning_rate * error


def _map_to_bounds(value: float, lower: float, upper: float) -> float:
    return lower + (upper - lower) * clamp(value, 0.0, 1.0)


def _serialize_rng_state(value: object) -> object:
    if isinstance(value, tuple):
        return [_serialize_rng_state(item) for item in value]
    if isinstance(value, list):
        return [_serialize_rng_state(item) for item in value]
    return value


def _deserialize_rng_state(value: object) -> object:
    if isinstance(value, list):
        return tuple(_deserialize_rng_state(item) for item in value)
    return value


def _serialize_categorical_head(head: CategoricalHead) -> dict[str, object]:
    return {
        "weights": [list(row) for row in head.weights],
        "bias": list(head.bias),
    }


def _categorical_head_shape(payload: object, *, label: str) -> tuple[int, int]:
    if not isinstance(payload, dict):
        raise TypeError(f"{label} payload must be a dict")
    weights = payload.get("weights")
    bias = payload.get("bias")
    if not isinstance(weights, list) or not isinstance(bias, list):
        raise TypeError(f"{label} payload must contain list weights and bias")
    if len(weights) != len(bias):
        raise ValueError(
            f"{label} payload has {len(weights)} rows but {len(bias)} bias values"
        )
    row_width: int | None = None
    for index, row in enumerate(weights):
        if not isinstance(row, list):
            raise TypeError(f"{label} row {index} must be a list")
        if row_width is None:
            row_width = len(row)
        elif len(row) != row_width:
            raise ValueError(f"{label} rows must all have the same width")
    return len(weights), (row_width or 0)


def _validate_categorical_head_payload(
    label: str,
    payload: object,
    *,
    expected_input_dim: int,
    expected_output_dim: int,
) -> None:
    output_dim, input_dim = _categorical_head_shape(payload, label=label)
    if input_dim != expected_input_dim or output_dim != expected_output_dim:
        raise ValueError(
            f"{label} checkpoint shape mismatch: expected "
            f"{expected_output_dim}x{expected_input_dim}, got {output_dim}x{input_dim}"
        )


def _validate_head_payload_sequence(
    label: str,
    payload: object,
    *,
    expected_count: int,
    expected_input_dim: int,
    expected_output_dim: int,
) -> None:
    if not isinstance(payload, list):
        raise TypeError(f"{label} payload must be a list")
    if len(payload) != expected_count:
        raise ValueError(
            f"{label} checkpoint count mismatch: expected {expected_count}, got {len(payload)}"
        )
    for index, item in enumerate(payload):
        _validate_categorical_head_payload(
            f"{label}[{index}]",
            item,
            expected_input_dim=expected_input_dim,
            expected_output_dim=expected_output_dim,
        )


def _finite_float(value: object, *, label: str) -> float:
    """Coerce a checkpoint-encoded number to ``float`` while rejecting NaN/Inf.

    Loading a JSON-encoded ``"Infinity"`` / ``"NaN"`` would otherwise inject
    poison values into policy weights and silently break training downstream;
    untrusted checkpoints can therefore use this to wedge a session.
    """
    return finite_float(value, label=label)


def _restore_categorical_head(head: CategoricalHead, payload: object) -> None:
    if not isinstance(payload, dict):
        raise TypeError("categorical head payload must be a dict")
    head.weights = [
        [
            _finite_float(value, label=f"weights[{i}][{j}]")
            for j, value in enumerate(row)
        ]
        for i, row in enumerate(payload["weights"])
    ]
    head.bias = [
        _finite_float(value, label=f"bias[{i}]")
        for i, value in enumerate(payload["bias"])
    ]


def _categorical_head_from_payload(payload: object) -> CategoricalHead:
    if not isinstance(payload, dict):
        raise TypeError("categorical head payload must be a dict")
    input_dim = len(payload["weights"][0]) if payload["weights"] else 0
    output_dim = len(payload["weights"])
    head = CategoricalHead(input_dim, output_dim, random.Random(0))
    _restore_categorical_head(head, payload)
    return head


def _serialize_gaussian_head(head: GaussianHead) -> dict[str, object]:
    return {
        "weights": [list(row) for row in head.weights],
        "bias": list(head.bias),
        "stddev": float(head.stddev),
    }


def _validate_gaussian_head_payload(
    label: str,
    payload: object,
    *,
    expected_input_dim: int,
    expected_output_dim: int,
) -> None:
    output_dim, input_dim = _categorical_head_shape(payload, label=label)
    if input_dim != expected_input_dim or output_dim != expected_output_dim:
        raise ValueError(
            f"{label} checkpoint shape mismatch: expected "
            f"{expected_output_dim}x{expected_input_dim}, got {output_dim}x{input_dim}"
        )
    if not isinstance(payload, dict):
        raise TypeError(f"{label} payload must be a dict")
    stddev = payload.get("stddev")
    if not isinstance(stddev, (int, float)) or isinstance(stddev, bool):
        raise TypeError(f"{label} stddev must be numeric")


def _gaussian_head_from_payload(payload: object) -> GaussianHead:
    if not isinstance(payload, dict):
        raise TypeError("gaussian head payload must be a dict")
    input_dim = len(payload["weights"][0]) if payload["weights"] else 0
    output_dim = len(payload["weights"])
    stddev = _finite_float(payload["stddev"], label="gaussian.stddev")
    if stddev < 0.0:
        raise ValueError(f"gaussian.stddev must be >= 0, got {stddev!r}")
    head = GaussianHead(input_dim, output_dim, random.Random(0), stddev)
    head.weights = [
        [
            _finite_float(value, label=f"gaussian.weights[{i}][{j}]")
            for j, value in enumerate(row)
        ]
        for i, row in enumerate(payload["weights"])
    ]
    head.bias = [
        _finite_float(value, label=f"gaussian.bias[{i}]")
        for i, value in enumerate(payload["bias"])
    ]
    return head


def _serialize_value_head(head: ValueHead) -> dict[str, object]:
    return {
        "weights": list(head.weights),
        "bias": float(head.bias),
    }


def _validate_value_head_payload(
    label: str, payload: object, *, expected_input_dim: int
) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"{label} payload must be a dict")
    weights = payload.get("weights")
    bias = payload.get("bias")
    if not isinstance(weights, list):
        raise TypeError(f"{label} weights must be a list")
    if len(weights) != expected_input_dim:
        raise ValueError(
            f"{label} checkpoint width mismatch: expected {expected_input_dim}, got {len(weights)}"
        )
    if not isinstance(bias, (int, float)) or isinstance(bias, bool):
        raise TypeError(f"{label} bias must be numeric")


def _value_head_from_payload(payload: object) -> ValueHead:
    if not isinstance(payload, dict):
        raise TypeError("value head payload must be a dict")
    head = ValueHead(len(payload["weights"]), random.Random(0))
    head.weights = [
        _finite_float(value, label=f"value.weights[{i}]")
        for i, value in enumerate(payload["weights"])
    ]
    head.bias = _finite_float(payload["bias"], label="value.bias")
    return head
