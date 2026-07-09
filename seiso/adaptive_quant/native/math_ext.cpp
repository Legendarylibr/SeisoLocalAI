#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using Array1D = py::array_t<double, py::array::c_style | py::array::forcecast>;

// ---------------------------------------------------------------------------
// Minimal SHA-256 (public-domain style) for deterministic_float parity
// ---------------------------------------------------------------------------

struct Sha256Ctx {
    std::uint64_t bitlen = 0;
    std::uint32_t state[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    std::uint8_t data[64]{};
    std::size_t datalen = 0;
};

constexpr std::uint32_t rotr(std::uint32_t value, std::uint32_t bits) {
    return (value >> bits) | (value << (32u - bits));
}

void sha256_transform(Sha256Ctx &ctx, const std::uint8_t data[64]) {
    static constexpr std::uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
        0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
        0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
        0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };
    std::uint32_t m[64];
    for (int i = 0, j = 0; i < 16; ++i, j += 4) {
        m[i] = (static_cast<std::uint32_t>(data[j]) << 24)
            | (static_cast<std::uint32_t>(data[j + 1]) << 16)
            | (static_cast<std::uint32_t>(data[j + 2]) << 8)
            | static_cast<std::uint32_t>(data[j + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        const std::uint32_t s0 = rotr(m[i - 15], 7) ^ rotr(m[i - 15], 18) ^ (m[i - 15] >> 3);
        const std::uint32_t s1 = rotr(m[i - 2], 17) ^ rotr(m[i - 2], 19) ^ (m[i - 2] >> 10);
        m[i] = m[i - 16] + s0 + m[i - 7] + s1;
    }
    std::uint32_t a = ctx.state[0];
    std::uint32_t b = ctx.state[1];
    std::uint32_t c = ctx.state[2];
    std::uint32_t d = ctx.state[3];
    std::uint32_t e = ctx.state[4];
    std::uint32_t f = ctx.state[5];
    std::uint32_t g = ctx.state[6];
    std::uint32_t h = ctx.state[7];
    for (int i = 0; i < 64; ++i) {
        const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const std::uint32_t ch = (e & f) ^ ((~e) & g);
        const std::uint32_t temp1 = h + S1 + ch + k[i] + m[i];
        const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temp2 = S0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    ctx.state[0] += a;
    ctx.state[1] += b;
    ctx.state[2] += c;
    ctx.state[3] += d;
    ctx.state[4] += e;
    ctx.state[5] += f;
    ctx.state[6] += g;
    ctx.state[7] += h;
}

void sha256_update(Sha256Ctx &ctx, const std::uint8_t *data, std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
        ctx.data[ctx.datalen++] = data[i];
        if (ctx.datalen == 64) {
            sha256_transform(ctx, ctx.data);
            ctx.bitlen += 512;
            ctx.datalen = 0;
        }
    }
}

std::array<std::uint8_t, 32> sha256_final(Sha256Ctx &ctx) {
    std::size_t i = ctx.datalen;
    if (ctx.datalen < 56) {
        ctx.data[i++] = 0x80;
        while (i < 56) {
            ctx.data[i++] = 0x00;
        }
    } else {
        ctx.data[i++] = 0x80;
        while (i < 64) {
            ctx.data[i++] = 0x00;
        }
        sha256_transform(ctx, ctx.data);
        std::memset(ctx.data, 0, 56);
    }
    ctx.bitlen += static_cast<std::uint64_t>(ctx.datalen) * 8u;
    ctx.data[63] = static_cast<std::uint8_t>(ctx.bitlen);
    ctx.data[62] = static_cast<std::uint8_t>(ctx.bitlen >> 8);
    ctx.data[61] = static_cast<std::uint8_t>(ctx.bitlen >> 16);
    ctx.data[60] = static_cast<std::uint8_t>(ctx.bitlen >> 24);
    ctx.data[59] = static_cast<std::uint8_t>(ctx.bitlen >> 32);
    ctx.data[58] = static_cast<std::uint8_t>(ctx.bitlen >> 40);
    ctx.data[57] = static_cast<std::uint8_t>(ctx.bitlen >> 48);
    ctx.data[56] = static_cast<std::uint8_t>(ctx.bitlen >> 56);
    sha256_transform(ctx, ctx.data);

    std::array<std::uint8_t, 32> hash{};
    for (int idx = 0; idx < 4; ++idx) {
        hash[idx] = (ctx.state[0] >> (24 - idx * 8)) & 0xffu;
        hash[idx + 4] = (ctx.state[1] >> (24 - idx * 8)) & 0xffu;
        hash[idx + 8] = (ctx.state[2] >> (24 - idx * 8)) & 0xffu;
        hash[idx + 12] = (ctx.state[3] >> (24 - idx * 8)) & 0xffu;
        hash[idx + 16] = (ctx.state[4] >> (24 - idx * 8)) & 0xffu;
        hash[idx + 20] = (ctx.state[5] >> (24 - idx * 8)) & 0xffu;
        hash[idx + 24] = (ctx.state[6] >> (24 - idx * 8)) & 0xffu;
        hash[idx + 28] = (ctx.state[7] >> (24 - idx * 8)) & 0xffu;
    }
    return hash;
}

std::array<std::uint8_t, 32> sha256_bytes(const std::string &text) {
    Sha256Ctx ctx;
    sha256_update(
        ctx,
        reinterpret_cast<const std::uint8_t *>(text.data()),
        text.size()
    );
    return sha256_final(ctx);
}

double deterministic_float(const std::string &key, double lower, double upper) {
    const auto digest = sha256_bytes(key);
    std::uint64_t bucket_int = 0;
    for (int i = 0; i < 8; ++i) {
        bucket_int = (bucket_int << 8) | digest[static_cast<std::size_t>(i)];
    }
    const double bucket =
        static_cast<double>(bucket_int) / static_cast<double>(0xFFFFFFFFFFFFFFFFull);
    return lower + (upper - lower) * bucket;
}

// ---------------------------------------------------------------------------
// Sequence / buffer helpers
// ---------------------------------------------------------------------------

std::vector<double> sequence_to_doubles(const py::sequence &values) {
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(py::len(values)));
    for (const py::handle item : values) {
        out.push_back(py::cast<double>(item));
    }
    return out;
}

std::vector<int> sequence_to_ints(const py::sequence &values) {
    std::vector<int> out;
    out.reserve(static_cast<std::size_t>(py::len(values)));
    for (const py::handle item : values) {
        out.push_back(py::cast<int>(item));
    }
    return out;
}

py::array_t<double> vector_to_array(const std::vector<double> &values) {
    py::array_t<double> out(static_cast<py::ssize_t>(values.size()));
    auto buf = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < values.size(); ++i) {
        buf(static_cast<py::ssize_t>(i)) = values[i];
    }
    return out;
}

std::vector<double> array_to_vector(const Array1D &values) {
    auto buf = values.unchecked<1>();
    std::vector<double> out(static_cast<std::size_t>(buf.shape(0)));
    for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
        out[static_cast<std::size_t>(i)] = buf(i);
    }
    return out;
}

double clamp_value(double value, double lower, double upper) {
    return std::max(lower, std::min(upper, value));
}

double mean_vec(const std::vector<double> &data) {
    if (data.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (const double value : data) {
        total += value;
    }
    return total / static_cast<double>(data.size());
}

double variance_vec(const std::vector<double> &data) {
    if (data.size() < 2) {
        return 0.0;
    }
    const double avg = mean_vec(data);
    double squared = 0.0;
    for (const double value : data) {
        const double delta = value - avg;
        squared += delta * delta;
    }
    return squared / static_cast<double>(data.size());
}

std::pair<double, double> mean_variance_vec(const std::vector<double> &data) {
    if (data.empty()) {
        return {0.0, 0.0};
    }
    const double avg = mean_vec(data);
    if (data.size() < 2) {
        return {avg, 0.0};
    }
    return {avg, variance_vec(data)};
}

double norm_vec(const std::vector<double> &data) {
    double total = 0.0;
    for (const double value : data) {
        total += value * value;
    }
    return std::sqrt(total);
}

std::vector<double> softmax_vec(const std::vector<double> &data) {
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

std::vector<double> matvec_flat(
    const std::vector<double> &weights,
    const std::vector<double> &bias,
    const std::vector<double> &state,
    int rows,
    int cols
) {
    if (rows < 0 || cols < 0) {
        throw py::value_error("rows and cols must be non-negative");
    }
    if (static_cast<int>(weights.size()) != rows * cols) {
        throw py::value_error("weights size does not match rows*cols");
    }
    if (static_cast<int>(bias.size()) != rows) {
        throw py::value_error("bias length does not match rows");
    }
    if (static_cast<int>(state.size()) != cols) {
        throw py::value_error("state length does not match cols");
    }
    std::vector<double> out(static_cast<std::size_t>(rows));
    for (int row = 0; row < rows; ++row) {
        double total = bias[static_cast<std::size_t>(row)];
        const std::size_t offset = static_cast<std::size_t>(row * cols);
        for (int col = 0; col < cols; ++col) {
            total += weights[offset + static_cast<std::size_t>(col)]
                * state[static_cast<std::size_t>(col)];
        }
        out[static_cast<std::size_t>(row)] = total;
    }
    return out;
}

void categorical_update_flat_inplace(
    std::vector<double> &weights,
    std::vector<double> &bias,
    const std::vector<double> &state,
    int rows,
    int cols,
    int selected_index,
    const std::vector<double> &probabilities,
    double advantage,
    double learning_rate
) {
    if (static_cast<int>(probabilities.size()) != rows) {
        throw py::value_error("probability length does not match rows");
    }
    if (selected_index < 0 || selected_index >= rows) {
        throw py::value_error("selected_index out of range");
    }
    for (int row = 0; row < rows; ++row) {
        const double coefficient = (
            (row == selected_index ? 1.0 : 0.0)
            - probabilities[static_cast<std::size_t>(row)]
        ) * advantage;
        const double scaled = learning_rate * coefficient;
        const std::size_t offset = static_cast<std::size_t>(row * cols);
        for (int col = 0; col < cols; ++col) {
            weights[offset + static_cast<std::size_t>(col)] +=
                scaled * state[static_cast<std::size_t>(col)];
        }
        bias[static_cast<std::size_t>(row)] += scaled;
    }
}

void gaussian_update_flat_inplace(
    std::vector<double> &weights,
    std::vector<double> &bias,
    const std::vector<double> &state,
    int rows,
    int cols,
    const std::vector<double> &raw_samples,
    const std::vector<double> &raw_means,
    double advantage,
    double learning_rate,
    double variance
) {
    if (
        static_cast<int>(raw_samples.size()) != rows ||
        static_cast<int>(raw_means.size()) != rows
    ) {
        throw py::value_error("sample/mean lengths do not match rows");
    }
    const double denom = std::max(variance, 1e-6);
    for (int row = 0; row < rows; ++row) {
        const double coefficient = (
            (
                raw_samples[static_cast<std::size_t>(row)]
                - raw_means[static_cast<std::size_t>(row)]
            )
            / denom
        ) * advantage;
        const double scaled = learning_rate * coefficient;
        const std::size_t offset = static_cast<std::size_t>(row * cols);
        for (int col = 0; col < cols; ++col) {
            weights[offset + static_cast<std::size_t>(col)] +=
                scaled * state[static_cast<std::size_t>(col)];
        }
        bias[static_cast<std::size_t>(row)] += scaled;
    }
}

void value_update_flat_inplace(
    std::vector<double> &weights,
    const std::vector<double> &state,
    double error,
    double learning_rate
) {
    if (weights.size() != state.size()) {
        throw py::value_error("weights and state vector lengths differ");
    }
    const double scaled = learning_rate * error;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights[i] += scaled * state[i];
    }
}

std::vector<double> nested_weights_to_flat(const py::sequence &weights, int &rows, int &cols) {
    rows = static_cast<int>(py::len(weights));
    if (rows == 0) {
        cols = 0;
        return {};
    }
    const py::sequence first = py::cast<py::sequence>(weights[0]);
    cols = static_cast<int>(py::len(first));
    std::vector<double> flat;
    flat.reserve(static_cast<std::size_t>(rows * cols));
    for (int row = 0; row < rows; ++row) {
        const py::sequence row_seq = py::cast<py::sequence>(weights[row]);
        if (static_cast<int>(py::len(row_seq)) != cols) {
            throw py::value_error("matrix rows must all have the same width");
        }
        for (int col = 0; col < cols; ++col) {
            flat.push_back(py::cast<double>(row_seq[col]));
        }
    }
    return flat;
}

py::list flat_weights_to_nested(const std::vector<double> &flat, int rows, int cols) {
    py::list out;
    for (int row = 0; row < rows; ++row) {
        py::list row_list;
        const std::size_t offset = static_cast<std::size_t>(row * cols);
        for (int col = 0; col < cols; ++col) {
            row_list.append(flat[offset + static_cast<std::size_t>(col)]);
        }
        out.append(row_list);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Existing sequence APIs (compat) + flat zero-copy APIs
// ---------------------------------------------------------------------------

py::list matrix_vector_add(
    const py::sequence &weights,
    const py::sequence &bias,
    const py::sequence &state_vector
) {
    int rows = 0;
    int cols = 0;
    const auto flat = nested_weights_to_flat(weights, rows, cols);
    const auto bias_vec = sequence_to_doubles(bias);
    const auto state = sequence_to_doubles(state_vector);
    const auto out = matvec_flat(flat, bias_vec, state, rows, cols);
    py::list result;
    for (const double value : out) {
        result.append(value);
    }
    return result;
}

Array1D matrix_vector_add_flat(
    const Array1D &weights,
    const Array1D &bias,
    const Array1D &state_vector,
    int rows,
    int cols
) {
    const auto out = matvec_flat(
        array_to_vector(weights),
        array_to_vector(bias),
        array_to_vector(state_vector),
        rows,
        cols
    );
    return vector_to_array(out);
}

double stable_sigmoid(double value) {
    if (value >= 0.0) {
        const double z = std::exp(-value);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(value);
    return z / (1.0 + z);
}

double mode_bonus_for(const std::string &mode) {
    if (mode == "discrete") {
        return 0.10;
    }
    if (mode == "grouped") {
        return 0.16;
    }
    if (mode == "per_layer") {
        return 0.18;
    }
    if (mode == "dynamic") {
        return 0.28;
    }
    if (mode == "learned") {
        return 0.34;
    }
    throw py::value_error("unsupported quantization mode");
}

std::vector<double> dynamic_layer_bits_vec(
    int base_bit_width,
    const std::vector<double> &layer_stats,
    double complexity,
    double min_bits,
    double max_bits
) {
    std::vector<double> out;
    out.reserve(layer_stats.size());
    for (const double layer_stat : layer_stats) {
        const double adjustment = 2.2 * (complexity - 0.45) + 1.7 * (layer_stat - 0.55);
        out.push_back(
            clamp_value(static_cast<double>(base_bit_width) + adjustment, min_bits, max_bits)
        );
    }
    return out;
}

std::vector<double> dynamic_layer_bits(
    int base_bit_width,
    const py::sequence &layer_stats,
    double complexity,
    double min_bits,
    double max_bits
) {
    return dynamic_layer_bits_vec(
        base_bit_width,
        sequence_to_doubles(layer_stats),
        complexity,
        min_bits,
        max_bits
    );
}

std::vector<double> learned_layer_bits_vec(
    const std::vector<double> &layer_stats,
    double precision_level,
    double precision_lower,
    double precision_upper,
    double precision_need,
    double scale_factor,
    double clipping_range,
    double min_bits,
    double max_bits
) {
    std::vector<double> out;
    out.reserve(layer_stats.size());
    const double learned_span = (max_bits - min_bits) * 0.75;
    const double base_bits =
        min_bits + clamp_value(precision_level, precision_lower, precision_upper) * learned_span;
    const std::size_t midpoint = layer_stats.size() / 2;
    for (std::size_t index = 0; index < layer_stats.size(); ++index) {
        const double layer_stat = layer_stats[index];
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
    return learned_layer_bits_vec(
        sequence_to_doubles(layer_stats),
        precision_level,
        precision_lower,
        precision_upper,
        precision_need,
        scale_factor,
        clipping_range,
        min_bits,
        max_bits
    );
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
    return mean_vec(sequence_to_doubles(values));
}

double variance(const py::sequence &values) {
    return variance_vec(sequence_to_doubles(values));
}

py::tuple mean_variance(const py::sequence &values) {
    const auto result = mean_variance_vec(sequence_to_doubles(values));
    return py::make_tuple(result.first, result.second);
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
    return norm_vec(sequence_to_doubles(values));
}

std::vector<double> softmax(const py::sequence &logits) {
    return softmax_vec(sequence_to_doubles(logits));
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
    const double avg = mean_vec(data);
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

void categorical_update_flat(
    Array1D weights,
    Array1D bias,
    const Array1D &state_vector,
    int rows,
    int cols,
    int selected_index,
    const Array1D &probabilities,
    double advantage,
    double learning_rate
) {
    auto w = array_to_vector(weights);
    auto b = array_to_vector(bias);
    categorical_update_flat_inplace(
        w,
        b,
        array_to_vector(state_vector),
        rows,
        cols,
        selected_index,
        array_to_vector(probabilities),
        advantage,
        learning_rate
    );
    auto wbuf = weights.mutable_unchecked<1>();
    auto bbuf = bias.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < wbuf.shape(0); ++i) {
        wbuf(i) = w[static_cast<std::size_t>(i)];
    }
    for (py::ssize_t i = 0; i < bbuf.shape(0); ++i) {
        bbuf(i) = b[static_cast<std::size_t>(i)];
    }
}

void gaussian_update_flat(
    Array1D weights,
    Array1D bias,
    const Array1D &state_vector,
    int rows,
    int cols,
    const Array1D &raw_samples,
    const Array1D &raw_means,
    double advantage,
    double learning_rate,
    double variance
) {
    auto w = array_to_vector(weights);
    auto b = array_to_vector(bias);
    gaussian_update_flat_inplace(
        w,
        b,
        array_to_vector(state_vector),
        rows,
        cols,
        array_to_vector(raw_samples),
        array_to_vector(raw_means),
        advantage,
        learning_rate,
        variance
    );
    auto wbuf = weights.mutable_unchecked<1>();
    auto bbuf = bias.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < wbuf.shape(0); ++i) {
        wbuf(i) = w[static_cast<std::size_t>(i)];
    }
    for (py::ssize_t i = 0; i < bbuf.shape(0); ++i) {
        bbuf(i) = b[static_cast<std::size_t>(i)];
    }
}

void value_update_flat(
    Array1D weights,
    const Array1D &state_vector,
    double error,
    double learning_rate
) {
    auto w = array_to_vector(weights);
    value_update_flat_inplace(w, array_to_vector(state_vector), error, learning_rate);
    auto wbuf = weights.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < wbuf.shape(0); ++i) {
        wbuf(i) = w[static_cast<std::size_t>(i)];
    }
}

py::tuple simulator_core_metrics(
    const std::string &mode,
    const std::string &hardware_type,
    double avg_bits,
    double bit_variance,
    double complexity,
    double sensitivity,
    double prompt_length,
    double latency_bias,
    double compute_factor,
    double throughput_bias,
    double kernel_uniformity_preference,
    double preferred_bits,
    double memory_budget_mb,
    double scale_factor,
    double clipping_range
) {
    const double mode_bonus = mode_bonus_for(mode);
    const double effective_prompt_length = std::max(8.0, prompt_length);
    const double compression = std::max(0.0, (8.0 - avg_bits) / 6.0);

    double latency_ms = (
        8.5
        * effective_prompt_length
        * latency_bias
        / std::max(0.35, compute_factor + (8.0 - avg_bits) * 0.12 + mode_bonus)
    );
    latency_ms *= (
        1.0
        + complexity * 0.55
        + std::max(0.0, bit_variance - kernel_uniformity_preference) * 0.18
    );

    double throughput_tps = (
        140.0
        * throughput_bias
        * (1.0 + (8.0 - avg_bits) * 0.10 + mode_bonus * 0.40)
        / (1.0 + complexity * 0.80 + latency_bias * 0.08)
    );
    if (hardware_type == "gpu") {
        throughput_tps *= 1.0 - std::min(0.12, bit_variance * 0.03);
    } else {
        throughput_tps *= 1.0 + std::min(0.10, std::max(0.0, preferred_bits - avg_bits) * 0.02);
    }

    double memory_mb = 4800.0 * (avg_bits / 16.0) * (1.0 + complexity * 0.15);
    if (mode == "per_layer" || mode == "learned") {
        memory_mb *= 1.02;
    }

    double perplexity = (
        5.6
        + complexity * 3.4
        + std::max(0.0, 5.5 - avg_bits) * (0.60 + complexity * 0.90 + sensitivity * 0.35)
        + std::abs(1.0 - scale_factor) * 0.65
        + std::max(0.0, 1.05 - clipping_range) * 1.20
        - mode_bonus * 0.70
    );

    const double hardware_alignment = std::abs(avg_bits - preferred_bits);
    latency_ms *= 1.0 + hardware_alignment * 0.04;
    throughput_tps *= 1.0 - hardware_alignment * 0.02;
    perplexity += hardware_alignment * 0.15;

    if ((hardware_type == "cpu" || hardware_type == "low_resource") && avg_bits > preferred_bits) {
        const double excess_bits = avg_bits - preferred_bits;
        latency_ms *= 1.0 + excess_bits * (hardware_type == "cpu" ? 0.16 : 0.24);
        throughput_tps *= std::max(0.55, 1.0 - excess_bits * (hardware_type == "cpu" ? 0.07 : 0.12));
        memory_mb *= 1.0 + excess_bits * (hardware_type == "cpu" ? 0.10 : 0.18);
    } else if (hardware_type == "gpu" && avg_bits < preferred_bits) {
        const double deficit_bits = preferred_bits - avg_bits;
        perplexity += deficit_bits * 0.45;
        throughput_tps *= std::max(0.78, 1.0 - deficit_bits * 0.03);
    }

    if (mode == "dynamic") {
        latency_ms *= 0.92;
        throughput_tps *= 1.06;
        perplexity -= 0.25 + complexity * 0.20;
    } else if (mode == "learned") {
        latency_ms *= 0.82 - compression * 0.06;
        throughput_tps *= 1.12 + compression * 0.08;
        memory_mb *= 0.78 - compression * 0.04;
        perplexity -= 0.38 + sensitivity * 0.22;
    } else if (mode == "grouped" && hardware_type != "gpu") {
        latency_ms *= 0.95;
        throughput_tps *= 1.03;
    }

    const double overflow_ratio = std::max(0.0, memory_mb - memory_budget_mb) / memory_budget_mb;
    if (overflow_ratio > 0.0) {
        latency_ms *= 1.0 + overflow_ratio * 2.50;
        throughput_tps *= 1.0 / (1.0 + overflow_ratio * 1.8);
        perplexity += overflow_ratio * 1.50;
    }

    return py::make_tuple(latency_ms, throughput_tps, perplexity, memory_mb);
}

double weighted_reward(
    double alpha_latency,
    double beta_throughput,
    double gamma_perplexity,
    double delta_memory,
    double epsilon_instability,
    double eta_token_latency,
    double zeta_perplexity_over_ref,
    double theta_kernel_speedup,
    double iota_kernel_latency,
    double latency_ms,
    double throughput_tps,
    double perplexity,
    double memory_mb,
    double latency_ms_per_token,
    double stability_penalty,
    bool include_instability,
    py::object perplexity_reference,
    double kernel_speedup,
    double kernel_latency_ms
) {
    double reward = (
        -alpha_latency * latency_ms
        + beta_throughput * throughput_tps
        - gamma_perplexity * perplexity
        - delta_memory * memory_mb
        - eta_token_latency * latency_ms_per_token
    );
    if (include_instability) {
        reward -= epsilon_instability * stability_penalty;
    }
    if (!perplexity_reference.is_none() && zeta_perplexity_over_ref > 0.0) {
        const double ref = py::cast<double>(perplexity_reference);
        reward -= zeta_perplexity_over_ref * std::max(0.0, perplexity - ref);
    }
    if (kernel_speedup > 0.0) {
        reward += theta_kernel_speedup * kernel_speedup;
    }
    if (kernel_latency_ms > 0.0) {
        reward -= iota_kernel_latency * kernel_latency_ms;
    }
    return reward;
}

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------

bool is_token_char(unsigned char ch) {
    return (ch >= 'a' && ch <= 'z')
        || (ch >= '0' && ch <= '9')
        || ch == '_'
        || (ch >= 'A' && ch <= 'Z');
}

std::vector<std::string> tokenize_text(const std::string &raw) {
    std::string text = raw;
    for (char &ch : text) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }
    std::vector<std::string> tokens;
    const std::size_t n = text.size();
    std::size_t i = 0;
    while (i < n) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        if (is_token_char(ch)) {
            const std::size_t start = i;
            ++i;
            while (i < n && is_token_char(static_cast<unsigned char>(text[i]))) {
                ++i;
            }
            tokens.emplace_back(text.substr(start, i - start));
        } else if (!std::isspace(ch)) {
            tokens.emplace_back(text.substr(i, 1));
            ++i;
        } else {
            ++i;
        }
    }
    return tokens;
}

py::list tokenize(const std::string &text) {
    py::list out;
    for (const auto &token : tokenize_text(text)) {
        out.append(token);
    }
    return out;
}

py::tuple extract_input_features(const std::string &text) {
    const auto tokens = tokenize_text(text);
    if (tokens.empty()) {
        return py::make_tuple(0, 0.0, 0.0, 0.0, 0.0);
    }

    std::unordered_map<std::string, int> counts;
    counts.reserve(tokens.size());
    for (const auto &token : tokens) {
        ++counts[token];
    }

    double entropy = 0.0;
    const double total = static_cast<double>(tokens.size());
    for (const auto &entry : counts) {
        const double probability = static_cast<double>(entry.second) / total;
        entropy -= probability * (std::log(probability + 1e-9) / std::log(2.0));
    }
    const double max_entropy =
        std::log(static_cast<double>(std::max(static_cast<int>(counts.size()), 2))) / std::log(2.0);
    const double normalized_entropy = max_entropy > 0.0 ? entropy / max_entropy : 0.0;

    std::vector<double> token_ids;
    token_ids.reserve(tokens.size());
    for (const auto &token : tokens) {
        token_ids.push_back(deterministic_float(token, 0.0, 1.0));
    }
    const double token_variance = clamp_value(variance_vec(token_ids) * 12.0, 0.0, 1.0);

    constexpr int embedding_dims = 8;
    std::vector<double> embedding(embedding_dims, 0.0);
    for (const auto &token : tokens) {
        for (int index = 0; index < embedding_dims; ++index) {
            embedding[static_cast<std::size_t>(index)] +=
                deterministic_float(token + ":" + std::to_string(index), -1.0, 1.0);
        }
    }
    for (double &value : embedding) {
        value /= static_cast<double>(tokens.size());
    }
    const double embedding_norm = clamp_value(
        norm_vec(embedding) / std::sqrt(static_cast<double>(embedding.size())),
        0.0,
        1.5
    );

    const double length_score = clamp_value(static_cast<double>(tokens.size()) / 80.0, 0.0, 1.4);
    const double complexity = clamp_value(
        0.35 * std::min(length_score, 1.0)
            + 0.30 * normalized_entropy
            + 0.20 * token_variance
            + 0.15 * std::min(embedding_norm, 1.0),
        0.0,
        1.25
    );
    return py::make_tuple(
        static_cast<int>(tokens.size()),
        normalized_entropy,
        token_variance,
        embedding_norm,
        complexity
    );
}

py::tuple estimate_layer_sensitivity(
    const std::string &prompt_id,
    const std::string &domain,
    double token_entropy,
    double token_variance,
    double embedding_norm,
    double complexity_score,
    int num_layers
) {
    if (num_layers < 0) {
        throw py::value_error("num_layers must be non-negative");
    }
    const double domain_bias = deterministic_float("domain:" + domain, -0.08, 0.08);
    const double attention = clamp_value(
        0.45 + 0.40 * token_entropy + 0.15 * complexity_score + domain_bias,
        0.0,
        1.4
    );
    const double ffn = clamp_value(
        0.42 + 0.38 * token_variance + 0.18 * embedding_norm + domain_bias,
        0.0,
        1.4
    );
    std::vector<double> layer_stats;
    layer_stats.reserve(static_cast<std::size_t>(num_layers));
    for (int layer_index = 0; layer_index < num_layers; ++layer_index) {
        const double phase = std::sin(
            (static_cast<double>(layer_index) + 1.0) * 0.8 + complexity_score * 2.2
        );
        const double layer_bias = deterministic_float(
            prompt_id + ":" + std::to_string(layer_index),
            -0.10,
            0.10
        );
        layer_stats.push_back(clamp_value(
            0.40
                + 0.25 * complexity_score
                + 0.15 * attention
                + 0.10 * ffn
                + 0.12 * phase
                + layer_bias,
            0.0,
            1.5
        ));
    }
    return py::make_tuple(attention, ffn, layer_stats);
}

double summarize_precision_needs(
    double complexity_score,
    double token_entropy,
    double token_variance,
    double attention_sensitivity,
    double ffn_sensitivity,
    const py::sequence &layer_stats
) {
    const auto stats = sequence_to_doubles(layer_stats);
    const std::vector<double> combined = {
        complexity_score,
        token_entropy,
        token_variance,
        attention_sensitivity,
        ffn_sensitivity,
        mean_vec(stats),
    };
    return clamp_value(mean_vec(combined), 0.0, 1.4);
}

// ---------------------------------------------------------------------------
// Decision finalize helpers
// ---------------------------------------------------------------------------

std::vector<double> pad_or_truncate_doubles(
    const std::vector<double> &values,
    int length,
    double fill
) {
    if (length <= 0) {
        return {};
    }
    if (static_cast<int>(values.size()) >= length) {
        return std::vector<double>(values.begin(), values.begin() + length);
    }
    std::vector<double> out = values;
    out.resize(static_cast<std::size_t>(length), fill);
    return out;
}

std::vector<int> pad_or_truncate_ints(
    const std::vector<int> &values,
    int length,
    int fill
) {
    if (length <= 0) {
        return {};
    }
    if (static_cast<int>(values.size()) >= length) {
        return std::vector<int>(values.begin(), values.begin() + length);
    }
    std::vector<int> out = values;
    out.resize(static_cast<std::size_t>(length), fill);
    return out;
}

std::vector<double> expand_group_bits(const py::sequence &group_bits, int num_layers) {
    const auto bits = sequence_to_ints(group_bits);
    if (bits.empty()) {
        return {};
    }
    const int layers_per_group = std::max(1, num_layers / static_cast<int>(bits.size()));
    std::vector<double> expanded;
    expanded.reserve(static_cast<std::size_t>(num_layers));
    for (const int bit_width : bits) {
        for (int i = 0; i < layers_per_group; ++i) {
            expanded.push_back(static_cast<double>(bit_width));
        }
    }
    return pad_or_truncate_doubles(
        expanded,
        num_layers,
        static_cast<double>(bits.back())
    );
}

int nearest_allowed_bit_width(
    py::object bit_width,
    const py::sequence &allowed,
    int default_bits
) {
    const auto allowed_bits = sequence_to_ints(allowed);
    if (allowed_bits.empty()) {
        return default_bits;
    }
    if (bit_width.is_none()) {
        return default_bits;
    }
    const int target = py::cast<int>(bit_width);
    int best = allowed_bits[0];
    int best_dist = std::abs(best - target);
    for (std::size_t i = 1; i < allowed_bits.size(); ++i) {
        const int candidate = allowed_bits[i];
        const int dist = std::abs(candidate - target);
        if (dist < best_dist) {
            best = candidate;
            best_dist = dist;
        }
    }
    return best;
}

py::list pad_or_truncate(const py::sequence &values, int length, py::object fill) {
    py::list out;
    if (length <= 0) {
        return out;
    }
    const auto size = static_cast<int>(py::len(values));
    const int copy_count = std::min(size, length);
    for (int i = 0; i < copy_count; ++i) {
        out.append(values[i]);
    }
    for (int i = copy_count; i < length; ++i) {
        out.append(fill);
    }
    return out;
}

py::tuple finalize_effective_layer_bits(
    const std::string &mode,
    int num_layers,
    py::object base_bit_width,
    const py::sequence &group_bit_widths,
    const py::sequence &layer_bit_widths,
    const py::sequence &allowed,
    int default_bits,
    const py::sequence &layer_stats,
    double complexity,
    double precision_level,
    double precision_lower,
    double precision_upper,
    double precision_need,
    double scale_factor,
    double clipping_range
) {
    const auto allowed_bits = sequence_to_ints(allowed);
    const int min_bits = allowed_bits.empty() ? default_bits : allowed_bits.front();
    const int max_bits = allowed_bits.empty() ? default_bits : allowed_bits.back();

    auto normalize = [&](py::object value) -> int {
        return nearest_allowed_bit_width(value, allowed, default_bits);
    };

    int out_base = default_bits;
    std::vector<int> out_group;
    std::vector<int> out_layer;
    std::vector<double> effective;

    if (mode == "discrete") {
        out_base = normalize(base_bit_width);
        effective.assign(static_cast<std::size_t>(num_layers), static_cast<double>(out_base));
    } else if (mode == "grouped") {
        for (const py::handle item : group_bit_widths) {
            out_group.push_back(normalize(py::reinterpret_borrow<py::object>(item)));
        }
        py::list group_list;
        for (const int bit : out_group) {
            group_list.append(bit);
        }
        effective = expand_group_bits(group_list, num_layers);
    } else if (mode == "per_layer") {
        if (py::len(layer_bit_widths) == 0) {
            out_layer.assign(static_cast<std::size_t>(num_layers), default_bits);
        } else {
            for (const py::handle item : layer_bit_widths) {
                out_layer.push_back(normalize(py::reinterpret_borrow<py::object>(item)));
            }
            const int fill = out_layer.empty() ? default_bits : out_layer.back();
            out_layer = pad_or_truncate_ints(out_layer, num_layers, fill);
        }
        effective.reserve(out_layer.size());
        for (const int bit : out_layer) {
            effective.push_back(static_cast<double>(bit));
        }
    } else if (mode == "dynamic") {
        out_base = normalize(base_bit_width);
        effective = dynamic_layer_bits_vec(
            out_base,
            sequence_to_doubles(layer_stats),
            complexity,
            static_cast<double>(min_bits),
            static_cast<double>(max_bits)
        );
    } else if (mode == "learned") {
        effective = learned_layer_bits_vec(
            sequence_to_doubles(layer_stats),
            precision_level,
            precision_lower,
            precision_upper,
            precision_need,
            scale_factor,
            clipping_range,
            static_cast<double>(min_bits),
            static_cast<double>(max_bits)
        );
    } else {
        throw py::value_error("unsupported quantization mode");
    }

    const auto stats = mean_variance_vec(effective);
    py::list group_py;
    for (const int bit : out_group) {
        group_py.append(bit);
    }
    py::list layer_py;
    for (const int bit : out_layer) {
        layer_py.append(bit);
    }
    return py::make_tuple(
        effective,
        stats.first,
        stats.second,
        out_base,
        group_py,
        layer_py
    );
}

// ---------------------------------------------------------------------------
// Native policy heads (weights stay in C++)
// ---------------------------------------------------------------------------

class FlatMatrixHead {
public:
    FlatMatrixHead(int rows, int cols)
        : rows_(rows), cols_(cols), weights_(static_cast<std::size_t>(rows * cols), 0.0),
          bias_(static_cast<std::size_t>(rows), 0.0) {
        if (rows < 0 || cols < 0) {
            throw py::value_error("rows and cols must be non-negative");
        }
    }

    int rows() const { return rows_; }
    int cols() const { return cols_; }

    py::list get_weights() const {
        return flat_weights_to_nested(weights_, rows_, cols_);
    }

    void set_weights(const py::sequence &weights) {
        int rows = 0;
        int cols = 0;
        auto flat = nested_weights_to_flat(weights, rows, cols);
        if (rows != rows_ || cols != cols_) {
            throw py::value_error("weight shape mismatch");
        }
        weights_ = std::move(flat);
    }

    py::list get_bias() const {
        py::list out;
        for (const double value : bias_) {
            out.append(value);
        }
        return out;
    }

    void set_bias(const py::sequence &bias) {
        if (static_cast<int>(py::len(bias)) != rows_) {
            throw py::value_error("bias length mismatch");
        }
        bias_ = sequence_to_doubles(bias);
    }

    std::vector<double> logits(const py::sequence &state_vector) const {
        return matvec_flat(weights_, bias_, sequence_to_doubles(state_vector), rows_, cols_);
    }

    void categorical_update(
        const py::sequence &state_vector,
        int selected_index,
        const py::sequence &probabilities,
        double advantage,
        double learning_rate
    ) {
        categorical_update_flat_inplace(
            weights_,
            bias_,
            sequence_to_doubles(state_vector),
            rows_,
            cols_,
            selected_index,
            sequence_to_doubles(probabilities),
            advantage,
            learning_rate
        );
    }

    void gaussian_update(
        const py::sequence &state_vector,
        const py::sequence &raw_samples,
        const py::sequence &raw_means,
        double advantage,
        double learning_rate,
        double variance
    ) {
        gaussian_update_flat_inplace(
            weights_,
            bias_,
            sequence_to_doubles(state_vector),
            rows_,
            cols_,
            sequence_to_doubles(raw_samples),
            sequence_to_doubles(raw_means),
            advantage,
            learning_rate,
            variance
        );
    }

private:
    int rows_;
    int cols_;
    std::vector<double> weights_;
    std::vector<double> bias_;
};

class FlatValueHead {
public:
    explicit FlatValueHead(int input_dim)
        : weights_(static_cast<std::size_t>(input_dim), 0.0), bias_(0.0) {
        if (input_dim < 0) {
            throw py::value_error("input_dim must be non-negative");
        }
    }

    py::list get_weights() const {
        py::list out;
        for (const double value : weights_) {
            out.append(value);
        }
        return out;
    }

    void set_weights(const py::sequence &weights) {
        if (static_cast<std::size_t>(py::len(weights)) != weights_.size()) {
            throw py::value_error("weight length mismatch");
        }
        weights_ = sequence_to_doubles(weights);
    }

    double get_bias() const { return bias_; }
    void set_bias(double bias) { bias_ = bias; }

    double predict(const py::sequence &state_vector) const {
        const auto state = sequence_to_doubles(state_vector);
        if (state.size() != weights_.size()) {
            throw py::value_error("state length mismatch");
        }
        double total = bias_;
        for (std::size_t i = 0; i < weights_.size(); ++i) {
            total += weights_[i] * state[i];
        }
        return total;
    }

    void update(const py::sequence &state_vector, double target, double learning_rate) {
        const auto state = sequence_to_doubles(state_vector);
        if (state.size() != weights_.size()) {
            throw py::value_error("state length mismatch");
        }
        double prediction = bias_;
        for (std::size_t i = 0; i < weights_.size(); ++i) {
            prediction += weights_[i] * state[i];
        }
        const double error = target - prediction;
        value_update_flat_inplace(weights_, state, error, learning_rate);
        bias_ += learning_rate * error;
    }

private:
    std::vector<double> weights_;
    double bias_;
};

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
    module.def(
        "matrix_vector_add_flat",
        &matrix_vector_add_flat,
        py::arg("weights"),
        py::arg("bias"),
        py::arg("state_vector"),
        py::arg("rows"),
        py::arg("cols")
    );
    module.def("stable_sigmoid", &stable_sigmoid, py::arg("value"));
    module.def("deterministic_float", &deterministic_float, py::arg("key"), py::arg("lower"), py::arg("upper"));
    module.def("tokenize", &tokenize, py::arg("text"));
    module.def("extract_input_features", &extract_input_features, py::arg("text"));
    module.def(
        "estimate_layer_sensitivity",
        &estimate_layer_sensitivity,
        py::arg("prompt_id"),
        py::arg("domain"),
        py::arg("token_entropy"),
        py::arg("token_variance"),
        py::arg("embedding_norm"),
        py::arg("complexity_score"),
        py::arg("num_layers")
    );
    module.def(
        "summarize_precision_needs",
        &summarize_precision_needs,
        py::arg("complexity_score"),
        py::arg("token_entropy"),
        py::arg("token_variance"),
        py::arg("attention_sensitivity"),
        py::arg("ffn_sensitivity"),
        py::arg("layer_stats")
    );
    module.def("expand_group_bits", &expand_group_bits, py::arg("group_bits"), py::arg("num_layers"));
    module.def(
        "pad_or_truncate",
        &pad_or_truncate,
        py::arg("values"),
        py::arg("length"),
        py::arg("fill")
    );
    module.def(
        "nearest_allowed_bit_width",
        &nearest_allowed_bit_width,
        py::arg("bit_width"),
        py::arg("allowed"),
        py::arg("default_bits")
    );
    module.def(
        "finalize_effective_layer_bits",
        &finalize_effective_layer_bits,
        py::arg("mode"),
        py::arg("num_layers"),
        py::arg("base_bit_width"),
        py::arg("group_bit_widths"),
        py::arg("layer_bit_widths"),
        py::arg("allowed"),
        py::arg("default_bits"),
        py::arg("layer_stats"),
        py::arg("complexity"),
        py::arg("precision_level"),
        py::arg("precision_lower"),
        py::arg("precision_upper"),
        py::arg("precision_need"),
        py::arg("scale_factor"),
        py::arg("clipping_range")
    );
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
    module.def(
        "categorical_update_flat",
        &categorical_update_flat,
        py::arg("weights"),
        py::arg("bias"),
        py::arg("state_vector"),
        py::arg("rows"),
        py::arg("cols"),
        py::arg("selected_index"),
        py::arg("probabilities"),
        py::arg("advantage"),
        py::arg("learning_rate")
    );
    module.def(
        "gaussian_update_flat",
        &gaussian_update_flat,
        py::arg("weights"),
        py::arg("bias"),
        py::arg("state_vector"),
        py::arg("rows"),
        py::arg("cols"),
        py::arg("raw_samples"),
        py::arg("raw_means"),
        py::arg("advantage"),
        py::arg("learning_rate"),
        py::arg("variance")
    );
    module.def(
        "value_update_flat",
        &value_update_flat,
        py::arg("weights"),
        py::arg("state_vector"),
        py::arg("error"),
        py::arg("learning_rate")
    );
    module.def(
        "simulator_core_metrics",
        &simulator_core_metrics,
        py::arg("mode"),
        py::arg("hardware_type"),
        py::arg("avg_bits"),
        py::arg("bit_variance"),
        py::arg("complexity"),
        py::arg("sensitivity"),
        py::arg("prompt_length"),
        py::arg("latency_bias"),
        py::arg("compute_factor"),
        py::arg("throughput_bias"),
        py::arg("kernel_uniformity_preference"),
        py::arg("preferred_bits"),
        py::arg("memory_budget_mb"),
        py::arg("scale_factor"),
        py::arg("clipping_range")
    );
    module.def(
        "weighted_reward",
        &weighted_reward,
        py::arg("alpha_latency"),
        py::arg("beta_throughput"),
        py::arg("gamma_perplexity"),
        py::arg("delta_memory"),
        py::arg("epsilon_instability"),
        py::arg("eta_token_latency"),
        py::arg("zeta_perplexity_over_ref"),
        py::arg("theta_kernel_speedup"),
        py::arg("iota_kernel_latency"),
        py::arg("latency_ms"),
        py::arg("throughput_tps"),
        py::arg("perplexity"),
        py::arg("memory_mb"),
        py::arg("latency_ms_per_token"),
        py::arg("stability_penalty"),
        py::arg("include_instability"),
        py::arg("perplexity_reference"),
        py::arg("kernel_speedup"),
        py::arg("kernel_latency_ms")
    );

    py::class_<FlatMatrixHead>(module, "FlatMatrixHead")
        .def(py::init<int, int>(), py::arg("rows"), py::arg("cols"))
        .def_property_readonly("rows", &FlatMatrixHead::rows)
        .def_property_readonly("cols", &FlatMatrixHead::cols)
        .def("get_weights", &FlatMatrixHead::get_weights)
        .def("set_weights", &FlatMatrixHead::set_weights)
        .def("get_bias", &FlatMatrixHead::get_bias)
        .def("set_bias", &FlatMatrixHead::set_bias)
        .def("logits", &FlatMatrixHead::logits)
        .def("categorical_update", &FlatMatrixHead::categorical_update)
        .def("gaussian_update", &FlatMatrixHead::gaussian_update);

    py::class_<FlatValueHead>(module, "FlatValueHead")
        .def(py::init<int>(), py::arg("input_dim"))
        .def("get_weights", &FlatValueHead::get_weights)
        .def("set_weights", &FlatValueHead::set_weights)
        .def("get_bias", &FlatValueHead::get_bias)
        .def("set_bias", &FlatValueHead::set_bias)
        .def("predict", &FlatValueHead::predict)
        .def("update", &FlatValueHead::update);
}
