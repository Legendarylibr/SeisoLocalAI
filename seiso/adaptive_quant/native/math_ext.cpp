#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

std::vector<double> sequence_to_doubles(const py::sequence &values) {
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(py::len(values)));
    for (const py::handle item : values) {
        out.push_back(py::cast<double>(item));
    }
    return out;
}

py::list matrix_vector_add(
    const py::sequence &weights,
    const py::sequence &bias,
    const py::sequence &state_vector
) {
    const auto rows = static_cast<py::ssize_t>(py::len(weights));
    const auto bias_len = static_cast<py::ssize_t>(py::len(bias));
    if (rows != bias_len) {
        throw py::value_error("weights row count and bias length differ");
    }
    const auto state = sequence_to_doubles(state_vector);
    py::list out;
    for (py::ssize_t row_index = 0; row_index < rows; ++row_index) {
        const py::sequence row = py::cast<py::sequence>(weights[row_index]);
        const auto row_len = static_cast<py::ssize_t>(py::len(row));
        if (row_len != static_cast<py::ssize_t>(state.size())) {
            throw py::value_error("matrix row length and state vector length differ");
        }
        double total = py::cast<double>(bias[row_index]);
        for (py::ssize_t column_index = 0; column_index < row_len; ++column_index) {
            total += py::cast<double>(row[column_index]) * state[static_cast<std::size_t>(column_index)];
        }
        out.append(total);
    }
    return out;
}

double stable_sigmoid(double value) {
    if (value >= 0.0) {
        const double z = std::exp(-value);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(value);
    return z / (1.0 + z);
}

double clamp_value(double value, double lower, double upper) {
    return std::max(lower, std::min(upper, value));
}

std::vector<double> dynamic_layer_bits(
    int base_bit_width,
    const py::sequence &layer_stats,
    double complexity,
    double min_bits,
    double max_bits
) {
    std::vector<double> out;
    const auto size = static_cast<py::ssize_t>(py::len(layer_stats));
    out.reserve(static_cast<std::size_t>(size));
    for (py::ssize_t index = 0; index < size; ++index) {
        const double layer_stat = py::cast<double>(layer_stats[index]);
        const double adjustment = 2.2 * (complexity - 0.45) + 1.7 * (layer_stat - 0.55);
        out.push_back(clamp_value(static_cast<double>(base_bit_width) + adjustment, min_bits, max_bits));
    }
    return out;
}

std::vector<double> learned_layer_bits(
    const py::sequence &layer_stats,
    double precision_level,
    double precision_lower,
    double precision_upper,
    double precision_need,
    double scale_factor,
    double clipping_range,
    double min_bits,
    double max_bits
) {
    const auto size = static_cast<py::ssize_t>(py::len(layer_stats));
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(size));
    const double learned_span = (max_bits - min_bits) * 0.75;
    const double base_bits = min_bits + clamp_value(precision_level, precision_lower, precision_upper) * learned_span;
    const py::ssize_t midpoint = size / 2;
    for (py::ssize_t index = 0; index < size; ++index) {
        const double layer_stat = py::cast<double>(layer_stats[index]);
        const double sensitivity_push = 1.05 * (layer_stat - 0.55) + 0.80 * (precision_need - 0.50);
        const double scale_push = (scale_factor - 1.0) * 0.45;
        const double clipping_push = (clipping_range - 1.0) * 0.35;
        const double depth_bias = index >= midpoint ? 0.12 : -0.04;
        out.push_back(clamp_value(
            base_bits + sensitivity_push + scale_push + clipping_push + depth_bias,
            min_bits,
            max_bits
        ));
    }
    return out;
}

py::tuple moe_variant_summary(
    const py::sequence &indices,
    int default_index,
    int variant_count
) {
    const auto size = static_cast<py::ssize_t>(py::len(indices));
    if (size == 0) {
        return py::make_tuple(0.0, 0.0);
    }
    const double denom = static_cast<double>(std::max(1, variant_count - 1));
    double aggressiveness_total = 0.0;
    double churn_total = 0.0;
    for (py::ssize_t index = 0; index < size; ++index) {
        const int variant_index = py::cast<int>(indices[index]);
        aggressiveness_total += static_cast<double>(variant_index) / denom;
        churn_total += std::abs(static_cast<double>(variant_index - default_index)) / denom;
    }
    const double count = static_cast<double>(size);
    return py::make_tuple(aggressiveness_total / count, churn_total / count);
}

double moe_swap_cost(
    const py::sequence &indices,
    const py::sequence &resident_on_device,
    const py::sequence &router_probabilities,
    const py::sequence &hotness,
    int variant_count
) {
    const auto size = static_cast<py::ssize_t>(py::len(indices));
    if (
        static_cast<py::ssize_t>(py::len(resident_on_device)) < size ||
        static_cast<py::ssize_t>(py::len(router_probabilities)) < size ||
        static_cast<py::ssize_t>(py::len(hotness)) < size
    ) {
        throw py::value_error("MoE swap cost inputs must have matching lengths");
    }
    const double denom = static_cast<double>(std::max(1, variant_count - 1));
    double total = 0.0;
    for (py::ssize_t index = 0; index < size; ++index) {
        const double resident = py::cast<double>(resident_on_device[index]);
        if (resident >= 0.5) {
            continue;
        }
        const double aggressiveness = static_cast<double>(py::cast<int>(indices[index])) / denom;
        const double probability = py::cast<double>(router_probabilities[index]);
        const double hot = py::cast<double>(hotness[index]);
        total += (1.2 + 3.4 * aggressiveness) * (0.75 + probability) * (1.10 - 0.35 * hot);
    }
    return total;
}

double mean(const py::sequence &values) {
    const auto data = sequence_to_doubles(values);
    if (data.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (const double value : data) {
        total += value;
    }
    return total / static_cast<double>(data.size());
}

double variance(const py::sequence &values) {
    const auto data = sequence_to_doubles(values);
    if (data.size() < 2) {
        return 0.0;
    }
    double total = 0.0;
    for (const double value : data) {
        total += value;
    }
    const double avg = total / static_cast<double>(data.size());
    double squared = 0.0;
    for (const double value : data) {
        const double delta = value - avg;
        squared += delta * delta;
    }
    return squared / static_cast<double>(data.size());
}

py::tuple mean_variance(const py::sequence &values) {
    const auto data = sequence_to_doubles(values);
    if (data.empty()) {
        return py::make_tuple(0.0, 0.0);
    }
    double total = 0.0;
    for (const double value : data) {
        total += value;
    }
    const double avg = total / static_cast<double>(data.size());
    if (data.size() < 2) {
        return py::make_tuple(avg, 0.0);
    }
    double squared = 0.0;
    for (const double value : data) {
        const double delta = value - avg;
        squared += delta * delta;
    }
    return py::make_tuple(avg, squared / static_cast<double>(data.size()));
}

double dot(const py::sequence &left, const py::sequence &right) {
    const auto left_len = static_cast<py::ssize_t>(py::len(left));
    const auto right_len = static_cast<py::ssize_t>(py::len(right));
    if (left_len != right_len) {
        throw py::value_error("zip() argument lengths differ");
    }
    double total = 0.0;
    for (py::ssize_t index = 0; index < left_len; ++index) {
        total += py::cast<double>(left[index]) * py::cast<double>(right[index]);
    }
    return total;
}

double norm(const py::sequence &values) {
    double total = 0.0;
    for (const py::handle item : values) {
        const double value = py::cast<double>(item);
        total += value * value;
    }
    return std::sqrt(total);
}

std::vector<double> softmax(const py::sequence &logits) {
    const auto data = sequence_to_doubles(logits);
    if (data.empty()) {
        return {};
    }
    const double max_logit = *std::max_element(data.begin(), data.end());
    std::vector<double> shifted;
    shifted.reserve(data.size());
    double total = 0.0;
    for (const double logit : data) {
        const double value = std::exp(logit - max_logit);
        shifted.push_back(value);
        total += value;
    }
    if (total <= 0.0) {
        return std::vector<double>(data.size(), 1.0 / static_cast<double>(data.size()));
    }
    for (double &value : shifted) {
        value /= total;
    }
    return shifted;
}

int argmax(const py::sequence &values) {
    const auto size = static_cast<py::ssize_t>(py::len(values));
    if (size == 0) {
        throw py::index_error("list index out of range");
    }
    int best_index = 0;
    double best_value = py::cast<double>(values[0]);
    for (py::ssize_t index = 1; index < size; ++index) {
        const double value = py::cast<double>(values[index]);
        if (value > best_value) {
            best_index = static_cast<int>(index);
            best_value = value;
        }
    }
    return best_index;
}

py::object safe_ratio(double numer, double denom) {
    if (!std::isfinite(numer) || !std::isfinite(denom) || denom <= 0.0) {
        return py::none();
    }
    return py::float_(numer / denom);
}

double ratio_mean(
    const py::sequence &observed,
    const py::sequence &simulated,
    double lower,
    double upper
) {
    double total = 0.0;
    std::size_t count = 0;
    const py::ssize_t size = std::min(py::len(observed), py::len(simulated));
    for (py::ssize_t index = 0; index < size; ++index) {
        const double o = py::cast<double>(observed[index]);
        const double s = py::cast<double>(simulated[index]);
        if (!std::isfinite(o) || !std::isfinite(s) || s <= 0.0) {
            continue;
        }
        const double ratio = o / s;
        if (lower < ratio && ratio < upper) {
            total += ratio;
            ++count;
        }
    }
    return count == 0 ? 1.0 : total / static_cast<double>(count);
}

double sample_std(const py::sequence &values) {
    const auto data = sequence_to_doubles(values);
    const std::size_t size = data.size();
    if (size < 2) {
        return 0.0;
    }
    double total = 0.0;
    for (const double value : data) {
        total += value;
    }
    const double avg = total / static_cast<double>(size);
    double squared = 0.0;
    for (const double value : data) {
        const double delta = value - avg;
        squared += delta * delta;
    }
    return std::sqrt(squared / static_cast<double>(size - 1));
}

void categorical_update(
    const py::sequence &weights,
    const py::sequence &bias,
    const py::sequence &state_vector,
    int selected_index,
    const py::sequence &probabilities,
    double advantage,
    double learning_rate
) {
    const auto rows = static_cast<py::ssize_t>(py::len(weights));
    if (
        static_cast<py::ssize_t>(py::len(bias)) != rows ||
        static_cast<py::ssize_t>(py::len(probabilities)) != rows
    ) {
        throw py::value_error("weights, bias, and probability lengths differ");
    }
    const auto state = sequence_to_doubles(state_vector);
    for (py::ssize_t row_index = 0; row_index < rows; ++row_index) {
        py::list row = py::cast<py::list>(weights[row_index]);
        if (static_cast<py::ssize_t>(py::len(row)) != static_cast<py::ssize_t>(state.size())) {
            throw py::value_error("matrix row length and state vector length differ");
        }
        const double coefficient = (
            (row_index == selected_index ? 1.0 : 0.0) - py::cast<double>(probabilities[row_index])
        ) * advantage;
        const double scaled = learning_rate * coefficient;
        for (py::ssize_t column_index = 0; column_index < static_cast<py::ssize_t>(state.size()); ++column_index) {
            const double current = py::cast<double>(row[column_index]);
            row[column_index] = current + scaled * state[static_cast<std::size_t>(column_index)];
        }
        const double bias_value = py::cast<double>(bias[row_index]);
        py::cast<py::list>(bias)[row_index] = bias_value + scaled;
    }
}

void gaussian_update(
    const py::sequence &weights,
    const py::sequence &bias,
    const py::sequence &state_vector,
    const py::sequence &raw_samples,
    const py::sequence &raw_means,
    double advantage,
    double learning_rate,
    double variance
) {
    const auto rows = static_cast<py::ssize_t>(py::len(weights));
    if (
        static_cast<py::ssize_t>(py::len(bias)) != rows ||
        static_cast<py::ssize_t>(py::len(raw_samples)) != rows ||
        static_cast<py::ssize_t>(py::len(raw_means)) != rows
    ) {
        throw py::value_error("weights, bias, sample, and mean lengths differ");
    }
    const auto state = sequence_to_doubles(state_vector);
    const double denom = std::max(variance, 1e-6);
    for (py::ssize_t row_index = 0; row_index < rows; ++row_index) {
        py::list row = py::cast<py::list>(weights[row_index]);
        if (static_cast<py::ssize_t>(py::len(row)) != static_cast<py::ssize_t>(state.size())) {
            throw py::value_error("matrix row length and state vector length differ");
        }
        const double coefficient = (
            (py::cast<double>(raw_samples[row_index]) - py::cast<double>(raw_means[row_index])) / denom
        ) * advantage;
        const double scaled = learning_rate * coefficient;
        for (py::ssize_t column_index = 0; column_index < static_cast<py::ssize_t>(state.size()); ++column_index) {
            const double current = py::cast<double>(row[column_index]);
            row[column_index] = current + scaled * state[static_cast<std::size_t>(column_index)];
        }
        const double bias_value = py::cast<double>(bias[row_index]);
        py::cast<py::list>(bias)[row_index] = bias_value + scaled;
    }
}

void value_update(
    const py::sequence &weights,
    const py::sequence &state_vector,
    double error,
    double learning_rate
) {
    const auto size = static_cast<py::ssize_t>(py::len(weights));
    if (static_cast<py::ssize_t>(py::len(state_vector)) != size) {
        throw py::value_error("weights and state vector lengths differ");
    }
    const double scaled = learning_rate * error;
    py::list weight_list = py::cast<py::list>(weights);
    for (py::ssize_t index = 0; index < size; ++index) {
        const double current = py::cast<double>(weight_list[index]);
        weight_list[index] = current + scaled * py::cast<double>(state_vector[index]);
    }
}

}  // namespace

PYBIND11_MODULE(_math_ext, module) {
    module.doc() = "Native math helpers for adaptive quantization hot paths.";
    module.def(
        "matrix_vector_add",
        &matrix_vector_add,
        py::arg("weights"),
        py::arg("bias"),
        py::arg("state_vector")
    );
    module.def("stable_sigmoid", &stable_sigmoid, py::arg("value"));
    module.def(
        "dynamic_layer_bits",
        &dynamic_layer_bits,
        py::arg("base_bit_width"),
        py::arg("layer_stats"),
        py::arg("complexity"),
        py::arg("min_bits"),
        py::arg("max_bits")
    );
    module.def(
        "learned_layer_bits",
        &learned_layer_bits,
        py::arg("layer_stats"),
        py::arg("precision_level"),
        py::arg("precision_lower"),
        py::arg("precision_upper"),
        py::arg("precision_need"),
        py::arg("scale_factor"),
        py::arg("clipping_range"),
        py::arg("min_bits"),
        py::arg("max_bits")
    );
    module.def(
        "moe_variant_summary",
        &moe_variant_summary,
        py::arg("indices"),
        py::arg("default_index"),
        py::arg("variant_count")
    );
    module.def(
        "moe_swap_cost",
        &moe_swap_cost,
        py::arg("indices"),
        py::arg("resident_on_device"),
        py::arg("router_probabilities"),
        py::arg("hotness"),
        py::arg("variant_count")
    );
    module.def("mean", &mean, py::arg("values"));
    module.def("variance", &variance, py::arg("values"));
    module.def("mean_variance", &mean_variance, py::arg("values"));
    module.def("dot", &dot, py::arg("left"), py::arg("right"));
    module.def("norm", &norm, py::arg("values"));
    module.def("softmax", &softmax, py::arg("logits"));
    module.def("argmax", &argmax, py::arg("values"));
    module.def("safe_ratio", &safe_ratio, py::arg("numer"), py::arg("denom"));
    module.def(
        "ratio_mean",
        &ratio_mean,
        py::arg("observed"),
        py::arg("simulated"),
        py::arg("lower") = 0.01,
        py::arg("upper") = 100.0
    );
    module.def("sample_std", &sample_std, py::arg("values"));
    module.def(
        "categorical_update",
        &categorical_update,
        py::arg("weights"),
        py::arg("bias"),
        py::arg("state_vector"),
        py::arg("selected_index"),
        py::arg("probabilities"),
        py::arg("advantage"),
        py::arg("learning_rate")
    );
    module.def(
        "gaussian_update",
        &gaussian_update,
        py::arg("weights"),
        py::arg("bias"),
        py::arg("state_vector"),
        py::arg("raw_samples"),
        py::arg("raw_means"),
        py::arg("advantage"),
        py::arg("learning_rate"),
        py::arg("variance")
    );
    module.def(
        "value_update",
        &value_update,
        py::arg("weights"),
        py::arg("state_vector"),
        py::arg("error"),
        py::arg("learning_rate")
    );
}
