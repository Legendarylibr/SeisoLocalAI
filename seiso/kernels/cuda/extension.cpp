#include "kernels.cuh"
#include "tuning_state.cuh"

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
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
        using scalar_t = __half;                                                      \
        return __VA_ARGS__();                                                         \
      }                                                                               \
      case at::ScalarType::BFloat16: {                                                \
        using scalar_t = __nv_bfloat16;                                               \
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
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  const void* res_ptr = residual.has_value() ? residual->data_ptr() : nullptr;

  DISPATCH_FLOAT_TYPES(x.scalar_type(), "fused_rmsnorm", [&] {
    using T = scalar_t;
    seiso::launch_rms_norm<T>(
        reinterpret_cast<const T*>(x.data_ptr()),
        residual.has_value() ? reinterpret_cast<const T*>(res_ptr) : nullptr,
        reinterpret_cast<const T*>(weight.data_ptr()),
        reinterpret_cast<T*>(out.data_ptr()),
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
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  DISPATCH_FLOAT_TYPES(gate.scalar_type(), "fused_swiglu", [&] {
    using T = scalar_t;
    seiso::launch_fused_swiglu<T>(
        reinterpret_cast<const T*>(gate.data_ptr()),
        reinterpret_cast<const T*>(up.data_ptr()),
        reinterpret_cast<T*>(out.data_ptr()),
        rows,
        cols,
        stream);
  });

  return out;
}

torch::Tensor fused_lora_delta(
    torch::Tensor x,
    torch::Tensor A,
    torch::Tensor B,
    c10::optional<torch::Tensor> base,
    double scale,
    bool inplace) {
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
    if (inplace) {
      out = *base;
    } else {
      out = torch::empty_like(*base);
    }
  } else if (x.dim() == 1) {
    out = torch::empty({out_dim}, x.options());
  } else {
    out = torch::empty({rows, out_dim}, x.options());
  }

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const float scale_f = static_cast<float>(scale);

  DISPATCH_FLOAT_TYPES(x.scalar_type(), "fused_lora_delta", [&] {
    using T = scalar_t;
    seiso::launch_fused_lora_delta<T>(
        reinterpret_cast<const T*>(x.data_ptr()),
        base.has_value() ? reinterpret_cast<const T*>(base->data_ptr()) : nullptr,
        reinterpret_cast<const T*>(A.data_ptr()),
        reinterpret_cast<const T*>(B.data_ptr()),
        reinterpret_cast<T*>(out.data_ptr()),
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
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  DISPATCH_FLOAT_TYPES(logits.scalar_type(), "cross_entropy_forward", [&] {
    using T = scalar_t;
    seiso::launch_cross_entropy_forward<T>(
        reinterpret_cast<const T*>(logits.data_ptr()),
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
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  DISPATCH_FLOAT_TYPES(logits.scalar_type(), "cross_entropy_backward", [&] {
    using T = scalar_t;
    seiso::launch_cross_entropy_backward<T>(
        reinterpret_cast<const T*>(logits.data_ptr()),
        labels.data_ptr<int64_t>(),
        row_max.data_ptr<float>(),
        row_lse.data_ptr<float>(),
        reinterpret_cast<T*>(grad_logits.data_ptr()),
        rows,
        vocab,
        static_cast<int>(ignore_index),
        static_cast<float>(grad_scale),
        stream);
  });

  return grad_logits;
}

void fused_lora_qkv_delta(
    torch::Tensor x,
    torch::Tensor out_q,
    torch::Tensor out_k,
    torch::Tensor out_v,
    torch::Tensor A_q,
    torch::Tensor B_q,
    torch::Tensor A_k,
    torch::Tensor B_k,
    torch::Tensor A_v,
    torch::Tensor B_v,
    double scale_q,
    double scale_k,
    double scale_v) {
  check_cuda(x, "x");
  check_cuda(out_q, "out_q");
  check_cuda(out_k, "out_k");
  check_cuda(out_v, "out_v");
  check_cuda(A_q, "A_q");
  check_cuda(B_q, "B_q");
  check_cuda(A_k, "A_k");
  check_cuda(B_k, "B_k");
  check_cuda(A_v, "A_v");
  check_cuda(B_v, "B_v");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(out_q.dim() == 2 && out_k.dim() == 2 && out_v.dim() == 2, "outputs must be 2D");
  TORCH_CHECK(A_q.dim() == 2 && B_q.dim() == 2, "A_q and B_q must be 2D");
  TORCH_CHECK(A_k.dim() == 2 && B_k.dim() == 2, "A_k and B_k must be 2D");
  TORCH_CHECK(A_v.dim() == 2 && B_v.dim() == 2, "A_v and B_v must be 2D");
  TORCH_CHECK(
      out_q.sizes() == out_k.sizes() && out_q.sizes() == out_v.sizes(),
      "out_q/out_k/out_v shape mismatch");
  TORCH_CHECK(out_q.size(0) == x.size(0), "batch mismatch");
  TORCH_CHECK(out_q.size(1) == B_q.size(0), "out_q out_dim mismatch");
  TORCH_CHECK(out_k.size(1) == B_k.size(0), "out_k out_dim mismatch");
  TORCH_CHECK(out_v.size(1) == B_v.size(0), "out_v out_dim mismatch");

  const int rows = static_cast<int>(x.size(0));
  const int in_dim = static_cast<int>(x.size(1));
  const int out_dim = static_cast<int>(out_q.size(1));
  const int rank = static_cast<int>(A_q.size(0));
  TORCH_CHECK(rank > 0 && rank <= 64, "rank must be in (0, 64]");
  TORCH_CHECK(A_q.size(1) == in_dim, "A_q in_dim mismatch");
  TORCH_CHECK(A_k.size(0) == rank && A_k.size(1) == in_dim, "A_k shape mismatch");
  TORCH_CHECK(A_v.size(0) == rank && A_v.size(1) == in_dim, "A_v shape mismatch");
  TORCH_CHECK(B_q.size(1) == rank, "B_q rank mismatch");
  TORCH_CHECK(B_k.size(1) == rank, "B_k rank mismatch");
  TORCH_CHECK(B_v.size(1) == rank, "B_v rank mismatch");
  const auto dtype = x.scalar_type();
  TORCH_CHECK(out_q.scalar_type() == dtype, "out_q dtype mismatch");
  TORCH_CHECK(out_k.scalar_type() == dtype, "out_k dtype mismatch");
  TORCH_CHECK(out_v.scalar_type() == dtype, "out_v dtype mismatch");
  TORCH_CHECK(A_q.scalar_type() == dtype, "A_q dtype mismatch");
  TORCH_CHECK(B_q.scalar_type() == dtype, "B_q dtype mismatch");
  TORCH_CHECK(A_k.scalar_type() == dtype, "A_k dtype mismatch");
  TORCH_CHECK(B_k.scalar_type() == dtype, "B_k dtype mismatch");
  TORCH_CHECK(A_v.scalar_type() == dtype, "A_v dtype mismatch");
  TORCH_CHECK(B_v.scalar_type() == dtype, "B_v dtype mismatch");
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  DISPATCH_FLOAT_TYPES(x.scalar_type(), "fused_lora_qkv_delta", [&] {
    using T = scalar_t;
    seiso::launch_fused_lora_qkv_delta<T>(
        reinterpret_cast<const T*>(x.data_ptr()),
        reinterpret_cast<T*>(out_q.data_ptr()),
        reinterpret_cast<T*>(out_k.data_ptr()),
        reinterpret_cast<T*>(out_v.data_ptr()),
        reinterpret_cast<const T*>(A_q.data_ptr()),
        reinterpret_cast<const T*>(B_q.data_ptr()),
        reinterpret_cast<const T*>(A_k.data_ptr()),
        reinterpret_cast<const T*>(B_k.data_ptr()),
        reinterpret_cast<const T*>(A_v.data_ptr()),
        reinterpret_cast<const T*>(B_v.data_ptr()),
        rows,
        in_dim,
        out_dim,
        rank,
        static_cast<float>(scale_q),
        static_cast<float>(scale_k),
        static_cast<float>(scale_v),
        stream);
  });
}

torch::Tensor fused_mlp_swiglu(
    torch::Tensor x,
    torch::Tensor W_gate,
    torch::Tensor W_up) {
  check_cuda(x, "x");
  check_cuda(W_gate, "W_gate");
  check_cuda(W_up, "W_up");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(W_gate.dim() == 2 && W_up.dim() == 2, "weights must be 2D");

  const int rows = static_cast<int>(x.size(0));
  const int in_dim = static_cast<int>(x.size(1));
  const int hidden_dim = static_cast<int>(W_gate.size(0));
  auto out = torch::empty({rows, hidden_dim}, x.options());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  DISPATCH_FLOAT_TYPES(x.scalar_type(), "fused_mlp_swiglu", [&] {
    using T = scalar_t;
    seiso::launch_fused_mlp_swiglu<T>(
        reinterpret_cast<const T*>(x.data_ptr()),
        reinterpret_cast<const T*>(W_gate.data_ptr()),
        reinterpret_cast<const T*>(W_up.data_ptr()),
        reinterpret_cast<T*>(out.data_ptr()),
        rows,
        in_dim,
        hidden_dim,
        stream);
  });
  return out;
}

void set_kernel_tuning(
    int rms_mode,
    int swiglu_vec,
    int lora_tile,
    int arch_sm,
    int use_cuda_graphs,
    int use_stream_overlap) {
  seiso::set_kernel_tuning_state(
      rms_mode, swiglu_vec, lora_tile, arch_sm, use_cuda_graphs, use_stream_overlap);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Seiso fused CUDA kernels for LLM training";
  m.def("fused_rmsnorm", &fused_rmsnorm, "Fused residual + RMSNorm");
  m.def("fused_swiglu", &fused_swiglu, "Fused SwiGLU activation");
  m.def("fused_lora_delta", &fused_lora_delta, "Fused low-rank LoRA delta");
  m.def("fused_lora_qkv_delta", &fused_lora_qkv_delta, "Fused LoRA QKV delta");
  m.def("fused_mlp_swiglu", &fused_mlp_swiglu, "Fused gate/up matmul + SwiGLU");
  m.def("cross_entropy_forward", &cross_entropy_forward, "Fused CE forward stats");
  m.def("cross_entropy_backward", &cross_entropy_backward, "Fused CE backward");
  m.def("set_kernel_tuning", &set_kernel_tuning, "Set RL kernel launch profile");
}
