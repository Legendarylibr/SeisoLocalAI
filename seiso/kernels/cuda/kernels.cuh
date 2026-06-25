#pragma once

#include <cstdint>

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

template <typename T>
void launch_fused_lora_qkv_delta(
    const T* x,
    T* out_q,
    T* out_k,
    T* out_v,
    const T* A_q,
    const T* B_q,
    const T* A_k,
    const T* B_k,
    const T* A_v,
    const T* B_v,
    int rows,
    int in_dim,
    int out_dim,
    int rank,
    float scale_q,
    float scale_k,
    float scale_v,
    cudaStream_t stream);

template <typename T>
void launch_fused_mlp_swiglu(
    const T* x,
    const T* W_gate,
    const T* W_up,
    T* out,
    int rows,
    int in_dim,
    int hidden_dim,
    cudaStream_t stream);

}  // namespace seiso
