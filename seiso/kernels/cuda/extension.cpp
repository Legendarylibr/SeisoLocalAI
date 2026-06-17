#include "kernels.cuh"

#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace {

void check_cuda(const torch::Tensor& t, const char* name) {
  if (!t.is_cuda()) {
    throw std::runtime_error(std::string(name) + " must be a CUDA tensor");
  }
  if (!t.is_contiguous()) {
    throw std::runtime_error(std::string(name) + " must be contiguous");
  }
}

#define DISPATCH_FLOAT_TYPES(TYPE, NAME, ...)                                         \
  [&] {                                                                               \
    const auto& the_type = (TYPE);                                                    \
    switch (the_type) {                                                               \
      case at::ScalarType::Float: {                                                   \
        using scalar_t = float;                                                       \
        return __VA_ARGS__();                                                         \
      }                                                                               \
      case at::ScalarType::Half: {                                                    \
        using scalar_t = at::Half;                                                    \
        return __VA_ARGS__();                                                         \
      }                                                                               \
      case at::ScalarType::BFloat16: {                                                \
        using scalar_t = at::BFloat16;                                                \
        return __VA_ARGS__();                                                         \
      }                                                                               \
      default:                                                                        \
        throw std::runtime_error(std::string("Unsupported dtype in ") + (NAME));      \
    }                                                                                 \
  }()

torch::Tensor fused_rmsnorm(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> residual,
    double eps) {
  check_cuda(x, "x");
  check_cuda(weight, "weight");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
  TORCH_CHECK(x.size(1) == weight.size(0), "hidden dim mismatch");
  if (residual.has_value()) {
    check_cuda(*residual, "residual");
    TORCH_CHECK(residual->sizes() == x.sizes(), "residual shape mismatch");
  }

  auto out = torch::empty_like(x);
  const int64_t rows = x.size(0);
  const int cols = static_cast<int>(x.size(1));
  const float eps_f = static_cast<float>(eps);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const void* res_ptr = residual.has_value() ? residual->data_ptr() : nullptr;

  DISPATCH_FLOAT_TYPES(x.scalar_type(), "fused_rmsnorm", [&] {
    using T = scalar_t;
    seiso::launch_rms_norm<T>(
        x.data_ptr<T>(),
        residual.has_value() ? static_cast<const T*>(res_ptr) : nullptr,
        weight.data_ptr<T>(),
        out.data_ptr<T>(),
        rows,
        cols,
        eps_f,
        residual.has_value(),
        stream);
  });

  return out;
}

torch::Tensor fused_swiglu(torch::Tensor gate, torch::Tensor up) {
  check_cuda(gate, "gate");
  check_cuda(up, "up");
  TORCH_CHECK(gate.sizes() == up.sizes(), "gate/up shape mismatch");
  TORCH_CHECK(gate.dim() == 2, "gate must be 2D");

  auto out = torch::empty_like(gate);
  const int64_t rows = gate.size(0);
  const int cols = static_cast<int>(gate.size(1));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_FLOAT_TYPES(gate.scalar_type(), "fused_swiglu", [&] {
    using T = scalar_t;
    seiso::launch_fused_swiglu<T>(
        gate.data_ptr<T>(), up.data_ptr<T>(), out.data_ptr<T>(), rows, cols, stream);
  });

  return out;
}

torch::Tensor fused_lora_delta(
    torch::Tensor x,
    torch::Tensor A,
    torch::Tensor B,
    c10::optional<torch::Tensor> base,
    double scale) {
  check_cuda(x, "x");
  check_cuda(A, "A");
  check_cuda(B, "B");
  TORCH_CHECK(x.dim() == 1 || x.dim() == 2, "x must be 1D or 2D");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");

  int rows = 1;
  int in_dim = 0;
  if (x.dim() == 1) {
    in_dim = static_cast<int>(x.size(0));
  } else {
    rows = static_cast<int>(x.size(0));
    in_dim = static_cast<int>(x.size(1));
  }

  const int rank = static_cast<int>(A.size(0));
  const int out_dim = static_cast<int>(B.size(0));
  TORCH_CHECK(A.size(1) == in_dim, "A in_dim mismatch");
  TORCH_CHECK(B.size(1) == rank, "B rank mismatch");
  TORCH_CHECK(rank > 0 && rank <= 64, "rank must be in (0, 64]");

  torch::Tensor out;
  if (base.has_value()) {
    check_cuda(*base, "base");
    TORCH_CHECK(base->sizes() == x.sizes() || (x.dim() == 1 && base->dim() == 1 && base->size(0) == out_dim),
                "base shape mismatch");
    out = torch::empty_like(*base);
  } else if (x.dim() == 1) {
    out = torch::empty({out_dim}, x.options());
  } else {
    out = torch::empty({rows, out_dim}, x.options());
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float scale_f = static_cast<float>(scale);

  DISPATCH_FLOAT_TYPES(x.scalar_type(), "fused_lora_delta", [&] {
    using T = scalar_t;
    seiso::launch_fused_lora_delta<T>(
        x.data_ptr<T>(),
        base.has_value() ? base->data_ptr<T>() : nullptr,
        A.data_ptr<T>(),
        B.data_ptr<T>(),
        out.data_ptr<T>(),
        rows,
        in_dim,
        out_dim,
        rank,
        scale_f,
        stream);
  });

  return out;
}

std::vector<torch::Tensor> cross_entropy_forward(
    torch::Tensor logits,
    torch::Tensor labels,
    int64_t ignore_index) {
  check_cuda(logits, "logits");
  check_cuda(labels, "labels");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  TORCH_CHECK(labels.dim() == 1, "labels must be 1D");
  TORCH_CHECK(logits.size(0) == labels.size(0), "batch mismatch");
  TORCH_CHECK(labels.scalar_type() == at::kLong, "labels must be int64");

  const int rows = static_cast<int>(logits.size(0));
  const int vocab = static_cast<int>(logits.size(1));
  auto opts = logits.options().dtype(at::kFloat);
  auto row_loss = torch::empty({rows}, opts);
  auto row_max = torch::empty({rows}, opts);
  auto row_lse = torch::empty({rows}, opts);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_FLOAT_TYPES(logits.scalar_type(), "cross_entropy_forward", [&] {
    using T = scalar_t;
    seiso::launch_cross_entropy_forward<T>(
        logits.data_ptr<T>(),
        labels.data_ptr<int64_t>(),
        row_loss.data_ptr<float>(),
        row_max.data_ptr<float>(),
        row_lse.data_ptr<float>(),
        rows,
        vocab,
        static_cast<int>(ignore_index),
        stream);
  });

  return {row_loss, row_max, row_lse};
}

torch::Tensor cross_entropy_backward(
    torch::Tensor logits,
    torch::Tensor labels,
    torch::Tensor row_max,
    torch::Tensor row_lse,
    int64_t ignore_index,
    double grad_scale) {
  check_cuda(logits, "logits");
  check_cuda(labels, "labels");
  auto grad_logits = torch::zeros_like(logits);
  const int rows = static_cast<int>(logits.size(0));
  const int vocab = static_cast<int>(logits.size(1));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_FLOAT_TYPES(logits.scalar_type(), "cross_entropy_backward", [&] {
    using T = scalar_t;
    seiso::launch_cross_entropy_backward<T>(
        logits.data_ptr<T>(),
        labels.data_ptr<int64_t>(),
        row_max.data_ptr<float>(),
        row_lse.data_ptr<float>(),
        grad_logits.data_ptr<T>(),
        rows,
        vocab,
        static_cast<int>(ignore_index),
        static_cast<float>(grad_scale),
        stream);
  });

  return grad_logits;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Seiso fused CUDA kernels for LLM training";
  m.def("fused_rmsnorm", &fused_rmsnorm, "Fused residual + RMSNorm");
  m.def("fused_swiglu", &fused_swiglu, "Fused SwiGLU activation");
  m.def("fused_lora_delta", &fused_lora_delta, "Fused low-rank LoRA delta");
  m.def("cross_entropy_forward", &cross_entropy_forward, "Fused CE forward stats");
  m.def("cross_entropy_backward", &cross_entropy_backward, "Fused CE backward");
}
