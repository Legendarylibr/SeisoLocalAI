#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace seiso {
namespace wmma_detail {

#if __CUDA_ARCH__ >= 700
#include <mma.h>

using namespace nvcuda::wmma;

// WMMA tile matmul: C[M,N] += A[M,K] @ B[K,N] with K=16 for fp16/bf16.
// Used for small-rank LoRA projections (rank <= 16).
template <typename T, int M, int N, int K>
__device__ __forceinline__ void mma_accum(
    float* c_acc,
    const T* a_tile,
    const T* b_tile) {
  fragment<matrix_a, M, N, K, T, row_major> a_frag;
  fragment<matrix_b, M, N, K, T, col_major> b_frag;
  fragment<accumulator, M, N, K, float> c_frag;

  load_matrix_sync(a_frag, a_tile, K);
  load_matrix_sync(b_frag, b_tile, K);
  fill_fragment(c_frag, 0.f);
  mma_sync(c_frag, a_frag, b_frag, c_frag);
  store_matrix_sync(c_acc, c_frag, N, mem_row_major);
}

// Dot a single row of A [K] with x [K] into acc[K] using WMMA when K==16.
template <typename T, int K>
__device__ __forceinline__ void lora_hidden_wmma(
    float* acc,
    const T* A_row_base,
    const T* x_vec,
    int in_dim,
    int rank) {
  if (rank != K) {
    return;
  }
  // Fallback scalar path handled by caller when rank != 16.
  T a_tile[K * K];
  T x_tile[K * K];
#pragma unroll
  for (int i = 0; i < K; ++i) {
#pragma unroll
    for (int j = 0; j < K; ++j) {
      const int col = j;
      a_tile[i * K + j] = (col < in_dim) ? A_row_base[static_cast<int64_t>(i) * in_dim + col] : T(0);
      x_tile[i * K + j] = (col < in_dim && i == 0) ? x_vec[col] : T(0);
    }
  }
  float c_tile[K * K];
  mma_accum<T, 16, 16, 16>(c_tile, a_tile, x_tile);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int r = 0; r < K; ++r) {
      acc[r] = c_tile[r];
    }
  }
}

#endif  // __CUDA_ARCH__ >= 700

}  // namespace wmma_detail
}  // namespace seiso