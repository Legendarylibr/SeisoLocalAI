#include "common.cuh"
#include "kernels.cuh"
#include "tuning_state.cuh"

#include <cstdint>
#include <type_traits>

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
  const auto& tuning = kernel_tuning_state();
  int tile = tuning.lora_tile > 0 ? tuning.lora_tile : 256;
  if (tile <= 128) {
    tile = 128;
  } else if (tile <= 256) {
    tile = 256;
  } else {
    tile = 512;
  }
  const dim3 grid((out_dim + tile - 1) / tile, static_cast<unsigned>(rows));

  auto launch = [&](auto tile_const, auto rank_const) {
    using Tile = decltype(tile_const);
    using Rank = decltype(rank_const);
    fused_lora_delta_batched_kernel<T, Tile::value, Rank::value>
        <<<grid, Tile::value, 0, stream>>>(
            x, base, A, B, out, rows, in_dim, out_dim, rank, scale);
  };

  if (tile == 128) {
    if (rank <= 16) {
      launch(std::integral_constant<int, 128>{}, std::integral_constant<int, 16>{});
    } else if (rank <= 32) {
      launch(std::integral_constant<int, 128>{}, std::integral_constant<int, 32>{});
    } else {
      launch(std::integral_constant<int, 128>{}, std::integral_constant<int, 64>{});
    }
  } else if (tile == 256) {
    if (rank <= 16) {
      launch(std::integral_constant<int, 256>{}, std::integral_constant<int, 16>{});
    } else if (rank <= 32) {
      launch(std::integral_constant<int, 256>{}, std::integral_constant<int, 32>{});
    } else {
      launch(std::integral_constant<int, 256>{}, std::integral_constant<int, 64>{});
    }
  } else {
    if (rank <= 16) {
      launch(std::integral_constant<int, 512>{}, std::integral_constant<int, 16>{});
    } else if (rank <= 32) {
      launch(std::integral_constant<int, 512>{}, std::integral_constant<int, 32>{});
    } else {
      launch(std::integral_constant<int, 512>{}, std::integral_constant<int, 64>{});
    }
  }
}

}  // namespace seiso

#include "explicit_inst.cuh"

namespace seiso {

template void launch_fused_lora_delta<float>(
    const float*, const float*, const float*, const float*, float*, int, int, int, int, float,
    cudaStream_t);
template void launch_fused_lora_delta<__half>(
    const __half*, const __half*, const __half*, const __half*, __half*, int, int, int, int, float,
    cudaStream_t);
template void launch_fused_lora_delta<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, int, int, int, int, float, cudaStream_t);

}  // namespace seiso
