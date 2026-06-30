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

double stable_sigmoid(double value) {
    if (value >= 0.0) {
        const double z = std::exp(-value);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(value);
    return z / (1.0 + z);
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

}  // namespace

PYBIND11_MODULE(_math_ext, module) {
    module.doc() = "Native math helpers for adaptive quantization hot paths.";
    module.def("stable_sigmoid", &stable_sigmoid, py::arg("value"));
    module.def("mean", &mean, py::arg("values"));
    module.def("variance", &variance, py::arg("values"));
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
}
