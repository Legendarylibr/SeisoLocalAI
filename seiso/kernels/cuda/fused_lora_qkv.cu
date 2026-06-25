/**
 * Fused LoRA QKV delta kernel.
 *
 * Computes in-place deltas for Q/K/V projections sharing a single input read:
 *   out_q += scale_q * B_q @ (A_q @ x)
 *   out_k += scale_k * B_k @ (A_k @ x)
 *   out_v += scale_v * B_v @ (A_v @ x)
 *
 * Features:
 *   - Persistent grid-stride row loop for steady-state throughput
 *   - Architecture-aware launch defaults via tuning state
 */

#include "arch_tuning.cuh"
#include "common.cuh"
#include "kernels.cuh"
#include "tuning_state.cuh"

#include <algorithm>
#include <cstdint>

namespace seiso {

constexpr int kQkvMaxRank = 64;
constexpr int kQkvBlock = 256;

template <typename T, int MAX_RANK, int TILE, bool PERSISTENT>
__global__ void __launch_bounds__(kQkvBlock)
    fused_lora_qkv_kernel(
        const T* __restrict__ x,
        T* __restrict__ out_q,
        T* __restrict__ out_k,
        T* __restrict__ out_v,
        const T* __restrict__ A_q,
        const T* __restrict__ B_q,
        const T* __restrict__ A_k,
        const T* __restrict__ B_k,
        const T* __restrict__ A_v,
        const T* __restrict__ B_v,
        int rows,
        int in_dim,
        int out_dim,
        int rank,
        float scale_q,
        float scale_k,
        float scale_v) {
  __shared__ float smem_hidden[3][MAX_RANK];

  const int tid = static_cast<int>(threadIdx.x);
  const int n_out_tiles = (out_dim + TILE - 1) / TILE;

  auto process_row = [&](int row) {
    if (row >= rows) {
      return;
    }

    const T* x_row = x + static_cast<int64_t>(row) * in_dim;
    const int64_t out_base = static_cast<int64_t>(row) * out_dim;

    // Phase 1: shared hidden = A @ x for Q, K, V (single-thread reduction)
    if (tid == 0) {
      float acc_q[MAX_RANK];
      float acc_k[MAX_RANK];
      float acc_v[MAX_RANK];
#pragma unroll
      for (int r = 0; r < MAX_RANK; ++r) {
        acc_q[r] = 0.f;
        acc_k[r] = 0.f;
        acc_v[r] = 0.f;
      }
      for (int k = 0; k < in_dim; ++k) {
        const float xv = to_float(x_row[k]);
#pragma unroll
        for (int r = 0; r < MAX_RANK; ++r) {
          if (r < rank) {
            const int64_t ar = static_cast<int64_t>(r) * in_dim + k;
            acc_q[r] += xv * to_float(A_q[ar]);
            acc_k[r] += xv * to_float(A_k[ar]);
            acc_v[r] += xv * to_float(A_v[ar]);
          }
        }
      }
#pragma unroll
      for (int r = 0; r < MAX_RANK; ++r) {
        smem_hidden[0][r] = (r < rank) ? acc_q[r] : 0.f;
        smem_hidden[1][r] = (r < rank) ? acc_k[r] : 0.f;
        smem_hidden[2][r] = (r < rank) ? acc_v[r] : 0.f;
      }
    }
    __syncthreads();

    // Phase 2: output tiles — each thread handles one out index per tile
    for (int tile = 0; tile < n_out_tiles; ++tile) {
      const int out_idx = tile * TILE + tid;
      if (out_idx >= out_dim) {
        continue;
      }

      float dq = 0.f;
      float dk = 0.f;
      float dv = 0.f;
#pragma unroll
      for (int r = 0; r < MAX_RANK; ++r) {
        if (r < rank) {
          const float bq = to_float(B_q[static_cast<int64_t>(out_idx) * rank + r]);
          const float bk = to_float(B_k[static_cast<int64_t>(out_idx) * rank + r]);
          const float bv = to_float(B_v[static_cast<int64_t>(out_idx) * rank + r]);
          dq += bq * smem_hidden[0][r];
          dk += bk * smem_hidden[1][r];
          dv += bv * smem_hidden[2][r];
        }
      }

      const int64_t off = out_base + out_idx;
      out_q[off] = from_float<T>(to_float(out_q[off]) + scale_q * dq);
      out_k[off] = from_float<T>(to_float(out_k[off]) + scale_k * dk);
      out_v[off] = from_float<T>(to_float(out_v[off]) + scale_v * dv);
    }
  };

  if (PERSISTENT) {
    const int grid_rows = static_cast<int>(gridDim.x);
    for (int row = static_cast<int>(blockIdx.x); row < rows; row += grid_rows) {
      process_row(row);
    }
  } else {
    process_row(static_cast<int>(blockIdx.x));
  }
}

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
    cudaStream_t stream) {
  const auto& tuning = kernel_tuning_state();
  const ArchLaunchDefaults defs = arch_launch_defaults(
      current_arch_family(), tuning.lora_tile > 0 ? tuning.lora_tile : 0);

  const bool persistent = defs.use_persistent && rows >= 4;

  if (rank <= 16) {
    if (persistent) {
      const int grid = min(rows, 65535);
      fused_lora_qkv_kernel<T, 16, 256, true>
          <<<grid, kQkvBlock, 0, stream>>>(
              x, out_q, out_k, out_v, A_q, B_q, A_k, B_k, A_v, B_v,
              rows, in_dim, out_dim, rank, scale_q, scale_k, scale_v);
    } else {
      fused_lora_qkv_kernel<T, 16, 256, false>
          <<<rows, kQkvBlock, 0, stream>>>(
              x, out_q, out_k, out_v, A_q, B_q, A_k, B_k, A_v, B_v,
              rows, in_dim, out_dim, rank, scale_q, scale_k, scale_v);
    }
  } else if (rank <= 32) {
    const int grid = persistent ? min(rows, 65535) : rows;
    if (persistent) {
      fused_lora_qkv_kernel<T, 32, 256, true>
          <<<grid, kQkvBlock, 0, stream>>>(
              x, out_q, out_k, out_v, A_q, B_q, A_k, B_k, A_v, B_v,
              rows, in_dim, out_dim, rank, scale_q, scale_k, scale_v);
    } else {
      fused_lora_qkv_kernel<T, 32, 256, false>
          <<<grid, kQkvBlock, 0, stream>>>(
              x, out_q, out_k, out_v, A_q, B_q, A_k, B_k, A_v, B_v,
              rows, in_dim, out_dim, rank, scale_q, scale_k, scale_v);
    }
  } else {
    const int grid = persistent ? min(rows, 65535) : rows;
    if (persistent) {
      fused_lora_qkv_kernel<T, 64, 256, true>
          <<<grid, kQkvBlock, 0, stream>>>(
              x, out_q, out_k, out_v, A_q, B_q, A_k, B_k, A_v, B_v,
              rows, in_dim, out_dim, rank, scale_q, scale_k, scale_v);
    } else {
      fused_lora_qkv_kernel<T, 64, 256, false>
          <<<grid, kQkvBlock, 0, stream>>>(
              x, out_q, out_k, out_v, A_q, B_q, A_k, B_k, A_v, B_v,
              rows, in_dim, out_dim, rank, scale_q, scale_k, scale_v);
    }
  }
}

}  // namespace seiso

#include "explicit_inst.cuh"

namespace seiso {

template void launch_fused_lora_qkv_delta<float>(
    const float*, float*, float*, float*, const float*, const float*, const float*, const float*,
    const float*, const float*, int, int, int, int, float, float, float, cudaStream_t);
template void launch_fused_lora_qkv_delta<__half>(
    const __half*, __half*, __half*, __half*, const __half*, const __half*, const __half*,
    const __half*, const __half*, const __half*, int, int, int, int, float, float, float,
    cudaStream_t);
template void launch_fused_lora_qkv_delta<__nv_bfloat16>(
    const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, int, int, int, int, float, float, float, cudaStream_t);

}  // namespace seiso