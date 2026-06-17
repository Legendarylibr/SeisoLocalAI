#include "common.cuh"
#include "kernels.cuh"

#include <cstdint>

namespace seiso {

// Fused low-rank delta: out = base + scale * B @ (A @ x)
// Supports batched rows: x [rows, in], out [rows, out], A [rank, in], B [out, rank]

template <typename T, int TILE, int MAX_RANK>
__global__ void fused_lora_delta_batched_kernel(
    const T* __restrict__ x,
    const T* __restrict__ base,
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ out,
    int rows,
    int in_dim,
    int out_dim,
    int rank,
    float scale) {
  __shared__ float smem_proj[MAX_RANK];

  const int row = static_cast<int>(blockIdx.y);
  const int out_idx = static_cast<int>(blockIdx.x) * TILE + static_cast<int>(threadIdx.x);
  if (row >= rows || out_idx >= out_dim) {
    return;
  }

  const T* x_row = x + static_cast<int64_t>(row) * in_dim;

  if (threadIdx.x == 0) {
    float acc[MAX_RANK];
#pragma unroll
    for (int r = 0; r < MAX_RANK; ++r) {
      acc[r] = 0.f;
    }
    for (int k = 0; k < in_dim; ++k) {
      float xv = to_float(x_row[k]);
#pragma unroll
      for (int r = 0; r < MAX_RANK; ++r) {
        if (r < rank) {
          acc[r] += xv * to_float(A[static_cast<int64_t>(r) * in_dim + k]);
        }
      }
    }
#pragma unroll
    for (int r = 0; r < MAX_RANK; ++r) {
      smem_proj[r] = (r < rank) ? acc[r] : 0.f;
    }
  }
  __syncthreads();

  float delta = 0.f;
#pragma unroll
  for (int r = 0; r < MAX_RANK; ++r) {
    if (r < rank) {
      delta += to_float(B[static_cast<int64_t>(out_idx) * rank + r]) * smem_proj[r];
    }
  }

  const int64_t out_offset = static_cast<int64_t>(row) * out_dim + out_idx;
  float base_v = base ? to_float(base[out_offset]) : 0.f;
  out[out_offset] = from_float<T>(base_v + scale * delta);
}

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
    cudaStream_t stream) {
  constexpr int TILE = 256;
  const dim3 grid((out_dim + TILE - 1) / TILE, static_cast<unsigned>(rows));
  if (rank <= 16) {
    fused_lora_delta_batched_kernel<T, TILE, 16><<<grid, TILE, 0, stream>>>(
        x, base, A, B, out, rows, in_dim, out_dim, rank, scale);
  } else if (rank <= 32) {
    fused_lora_delta_batched_kernel<T, TILE, 32><<<grid, TILE, 0, stream>>>(
        x, base, A, B, out, rows, in_dim, out_dim, rank, scale);
  } else {
    fused_lora_delta_batched_kernel<T, TILE, 64><<<grid, TILE, 0, stream>>>(
        x, base, A, B, out, rows, in_dim, out_dim, rank, scale);
  }
}

}  // namespace seiso
