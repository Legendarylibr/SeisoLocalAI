/**
 * Seiso "Stripe" RMSNorm — native CUDA with warp-stripe reduction.
 *
 * Design choices:
 *   - One CTA per row; columns striped across WARPS (not threads) for coalescing.
 *   - Variance reduced via warp-shuffles only — no block-wide shared-memory tree.
 *   - Optional fused residual add in the same pass (no extra global read/write).
 *   - cp.async 2-stage prefetch on sm_80+ to hide DRAM latency on wide hidden dims.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>

#include "include/seiso_vec.cuh"
#include "tuning_state.cuh"

namespace seiso {

constexpr int kWarpsPerBlock = 8;
constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;  // 256
constexpr int kElemsPerThread = 4;  // unrolled scalar chunks per inner step

// Always compile cp.async path in .cu TUs; launch targets sm_80+ via -arch=sm_XX.
#define SEISO_HAS_CP_ASYNC 1
#include <cuda_pipeline.h>

// ---------------------------------------------------------------------------
// Stripe RMSNorm (+ optional residual)
// ---------------------------------------------------------------------------

template <typename T, bool FUSE_RESIDUAL>
__global__ void __launch_bounds__(kThreadsPerBlock)
    stripe_rms_norm_kernel(
        const T* __restrict__ x,
        const T* __restrict__ residual,  // ignored when FUSE_RESIDUAL == false
        const T* __restrict__ weight,
        T* __restrict__ out,
        int64_t cols,
        float eps) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  const T* x_row = x + row * cols;
  const T* r_row = FUSE_RESIDUAL ? residual + row * cols : nullptr;
  T* o_row = out + row * cols;

  // Each warp owns a contiguous column stripe.
  const int64_t stripe_elems = (cols + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int64_t col_begin = static_cast<int64_t>(warp_id) * stripe_elems;
  const int64_t col_end = min(col_begin + stripe_elems, cols);

  float local_sq = 0.f;

  // Pass 1 — accumulate x^2 (and fuse residual add into the value we normalize)
  for (int64_t base = col_begin + lane * kElemsPerThread; base < col_end;
       base += kWarpSize * kElemsPerThread) {
    Vec<T, kElemsPerThread> vx;
    load_vec<T, kElemsPerThread>(x_row, static_cast<int>(base), static_cast<int>(cols), vx);

    if (FUSE_RESIDUAL) {
#pragma unroll
      for (int i = 0; i < kElemsPerThread; ++i) {
        const int j = static_cast<int>(base) + i;
        if (j < col_end) {
          const float xf = to_float(vx.data[i]);
          const float rf = to_float(r_row[j]);
          const float fused = xf + rf;
          local_sq += fused * fused;
        }
      }
    } else {
      local_sq += vec_sum_sq(vx);
    }
  }

  // Warp-local partial sum -> broadcast inverse RMS across warp
  const float warp_sq = warp_reduce_sum(local_sq);
  const float inv_rms = rsqrtf(warp_sq / static_cast<float>(cols) + eps);

  // Pass 2 — write normalized output (reuse fused value if residual)
  for (int64_t base = col_begin + lane * kElemsPerThread; base < col_end;
       base += kWarpSize * kElemsPerThread) {
    Vec<T, kElemsPerThread> vx;
    load_vec<T, kElemsPerThread>(x_row, static_cast<int>(base), static_cast<int>(cols), vx);

    Vec<T, kElemsPerThread> vw;
    load_vec<T, kElemsPerThread>(weight, static_cast<int>(base), static_cast<int>(cols), vw);

    Vec<T, kElemsPerThread> vo;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      const int j = static_cast<int>(base) + i;
      if (j < col_end) {
        float val = to_float(vx.data[i]);
        if (FUSE_RESIDUAL) {
          val += to_float(r_row[j]);
        }
        val = val * inv_rms * to_float(vw.data[i]);
        vo.data[i] = from_float<T>(val);
      }
    }
    store_vec<T, kElemsPerThread>(o_row, static_cast<int>(base), static_cast<int>(cols), vo);
  }
}

// Wide-hidden variant: double-buffered cp.async tile pipeline (Ampere+)
#if SEISO_HAS_CP_ASYNC

template <typename T, int TILE_ELEMS, bool FUSE_RESIDUAL>
__global__ void __launch_bounds__(kThreadsPerBlock)
    parallax_rms_norm_kernel(
        const T* __restrict__ x,
        const T* __restrict__ residual,
        const T* __restrict__ weight,
        T* __restrict__ out,
        int64_t cols,
        float eps) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;

  const T* x_row = x + row * cols;
  const T* r_row = FUSE_RESIDUAL ? residual + row * cols : nullptr;
  T* o_row = out + row * cols;

  const int64_t stripe_elems = (cols + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int64_t col_begin = static_cast<int64_t>(warp_id) * stripe_elems;
  const int64_t col_end = min(col_begin + stripe_elems, cols);

  constexpr int kBytesPerTile = TILE_ELEMS * static_cast<int>(sizeof(T));
  alignas(16) __shared__ char smem[2][kWarpsPerBlock][TILE_ELEMS * sizeof(T)];

  float local_sq = 0.f;

  const int64_t tile_stride = static_cast<int64_t>(kWarpsPerBlock) * TILE_ELEMS;
  int64_t tile_base = col_begin + static_cast<int64_t>(warp_id) * TILE_ELEMS;

  // Prefetch first tile
  if (tile_base < col_end) {
    const int dst = (tid & 1);
    const T* src = x_row + tile_base + (tid % TILE_ELEMS);
    if (tile_base + (tid % TILE_ELEMS) < col_end && (tid / TILE_ELEMS) == 0) {
      __pipeline_memcpy_async(smem[dst][warp_id], src, sizeof(T));
    }
    __pipeline_commit();
  }

  for (; tile_base < col_end; tile_base += tile_stride) {
    const int buf = (tile_base / tile_stride) & 1;
    __pipeline_wait_prior(0);
    __syncthreads();

    const T* tile = reinterpret_cast<const T*>(smem[buf][warp_id]);
#pragma unroll
    for (int i = lane; i < TILE_ELEMS; i += kWarpSize) {
      const int64_t j = tile_base + i;
      if (j < col_end) {
        float val = to_float(tile[i]);
        if (FUSE_RESIDUAL) {
          val += to_float(r_row[j]);
        }
        local_sq += val * val;
      }
    }

    const int64_t next_base = tile_base + tile_stride;
    if (next_base < col_end) {
      const int dst = 1 - buf;
      const T* src = x_row + next_base + (tid % TILE_ELEMS);
      if (next_base + (tid % TILE_ELEMS) < col_end && (tid / TILE_ELEMS) == 0) {
        __pipeline_memcpy_async(smem[dst][warp_id], src, sizeof(T));
      }
      __pipeline_commit();
    }
  }

  const float warp_sq = warp_reduce_sum(local_sq);
  const float inv_rms = rsqrtf(warp_sq / static_cast<float>(cols) + eps);

  // Second pass: straightforward vectorized (compute-bound, no async needed)
  for (int64_t base = col_begin + lane * kElemsPerThread; base < col_end;
       base += kWarpSize * kElemsPerThread) {
    Vec<T, kElemsPerThread> vx;
    load_vec<T, kElemsPerThread>(x_row, static_cast<int>(base), static_cast<int>(cols), vx);
    Vec<T, kElemsPerThread> vw;
    load_vec<T, kElemsPerThread>(weight, static_cast<int>(base), static_cast<int>(cols), vw);
    Vec<T, kElemsPerThread> vo;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      const int j = static_cast<int>(base) + i;
      if (j < col_end) {
        float val = to_float(vx.data[i]);
        if (FUSE_RESIDUAL) {
          val += to_float(r_row[j]);
        }
        val = val * inv_rms * to_float(vw.data[i]);
        vo.data[i] = from_float<T>(val);
      }
    }
    store_vec<T, kElemsPerThread>(o_row, static_cast<int>(base), static_cast<int>(cols), vo);
  }
}

#endif  // SEISO_HAS_CP_ASYNC

// ---------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------

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
    cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(rows));
  const dim3 block(kThreadsPerBlock);

  const auto& tuning = kernel_tuning_state();
  const bool wide = cols >= 4096;
  bool use_parallax = wide;
  if (tuning.rms_mode == RMS_STRIPE) {
    use_parallax = false;
  } else if (tuning.rms_mode == RMS_PARALLAX) {
    use_parallax = true;
  }

#if SEISO_HAS_CP_ASYNC
  // Parallax cp.async path disabled: pipeline memcpy requires 4-byte chunks and
  // the current prototype trips cudaErrorLaunchFailure on Ada/Hopper. Fall back
  // to stripe until the async prefetch kernel is reworked.
  (void)use_parallax;
#endif

  if (fuse_residual) {
    stripe_rms_norm_kernel<T, true><<<grid, block, 0, stream>>>(
        x, residual, weight, out, cols, eps);
  } else {
    stripe_rms_norm_kernel<T, false><<<grid, block, 0, stream>>>(
        x, residual, weight, out, cols, eps);
  }
}

}  // namespace seiso

#include "explicit_inst.cuh"

namespace seiso {

template void launch_rms_norm<float>(
    const float*, const float*, const float*, float*, int64_t, int64_t, float, bool, cudaStream_t);
template void launch_rms_norm<__half>(
    const __half*, const __half*, const __half*, __half*, int64_t, int64_t, float, bool,
    cudaStream_t);
template void launch_rms_norm<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int64_t,
    int64_t, float, bool, cudaStream_t);

}  // namespace seiso
