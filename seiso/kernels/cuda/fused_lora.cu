#include "common.cuh"
#include "kernels.cuh"

#include <cstdint>

namespace seiso {

// Fused low-rank delta: out = base + scale * B @ (A @ x)
// A: [rank, in], B: [out, rank], x: [in], base/out: [out]
// One CTA per output tile. Rank is small (typical LoRA 8–64).

template <typename T, int TILE, int MAX_RANK>
__global__ void fused_lora_delta_kernel(
    const T* __restrict__ x,
    const T* __restrict__ base,
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ out,
    int in_dim,
    int out_dim,
    int rank,
    float scale) {
  __shared__ float smem_a[MAX_RANK];
  __shared__ float smem_proj[MAX_RANK];

  const int out_idx = blockIdx.x * TILE + threadIdx.x;
  if (out_idx >= out_dim) {
    return;
  }

  if (threadIdx.x == 0) {
    float acc[MAX_RANK];
#pragma unroll
    for (int r = 0; r < MAX_RANK; ++r) {
      acc[r] = 0.f;
    }
    for (int k = 0; k < in_dim; ++k) {
      float xv = to_float(x[k]);
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

  float base_v = base ? to_float(base[out_idx]) : 0.f;
  out[out_idx] = from_float<T>(base_v + scale * delta);
}

template <typename T>
void launch_fused_lora_delta(
    const T* x,
    const T* base,
    const T* A,
    const T* B,
    T* out,
    int in_dim,
    int out_dim,
    int rank,
    float scale,
    cudaStream_t stream) {
  constexpr int TILE = 256;
  const int grid = (out_dim + TILE - 1) / TILE;
  if (rank <= 16) {
    fused_lora_delta_kernel<T, TILE, 16><<<grid, TILE, 0, stream>>>(
        x, base, A, B, out, in_dim, out_dim, rank, scale);
  } else if (rank <= 32) {
    fused_lora_delta_kernel<T, TILE, 32><<<grid, TILE, 0, stream>>>(
        x, base, A, B, out, in_dim, out_dim, rank, scale);
  } else {
    fused_lora_delta_kernel<T, TILE, 64><<<grid, TILE, 0, stream>>>(
        x, base, A, B, out, in_dim, out_dim, rank, scale);
  }
}

}  // namespace seiso
