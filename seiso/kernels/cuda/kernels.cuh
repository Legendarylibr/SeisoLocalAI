#pragma once

#include <cuda_runtime.h>

namespace seiso {

template <typename T>
void launch_rms_norm(
    const T* x,
    const T* residual,
    const T* weight,
    T* out,
    int64_t rows,
    int64_t cols,
    float eps,
    bool fuse_residual,
    cudaStream_t stream);

template <typename T>
void launch_fused_swiglu(
    const T* gate, const T* up, T* out, int64_t rows, int cols, cudaStream_t stream);

template <typename T>
void launch_fused_lora_delta(
    const T* x,
    const T* base,
    const T* A,
    const T* B,
    T* out,
    int rows,
    int in_dim,
    int out_dim,
    int rank,
    float scale,
    cudaStream_t stream);

template <typename T>
void launch_cross_entropy_forward(
    const T* logits,
    const int64_t* labels,
    float* row_loss,
    float* row_max,
    float* row_lse,
    int rows,
    int vocab,
    int ignore_index,
    cudaStream_t stream);

template <typename T>
void launch_cross_entropy_backward(
    const T* logits,
    const int64_t* labels,
    const float* row_max,
    const float* row_lse,
    T* grad_logits,
    int rows,
    int vocab,
    int ignore_index,
    float inv_count,
    cudaStream_t stream);

}  // namespace seiso
