/**
 * Seiso "Stripe" RMSNorm — native CUDA with warp-stripe reduction.
 *
 * Design choices:
 *   - One CTA per row; columns striped across WARPS for coalescing.
 *   - Variance reduced via warp-shuffles only within each warp's stripe pass,
 *     then a block-wide sum of squares (shared + warp) so inv_rms is correct
 *     for the full hidden dim.
 *   - Optional fused residual add in the same pass (no extra global read/write).
 *
 * Note: the former cp.async "parallax" path was removed (P2 cleanup) — it never
 * launched reliably on Ada/Hopper and RMS_PARALLAX now maps to stripe.
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

  // Warp-local partial → full-row sum of squares (block-wide).
  const float warp_sq = warp_reduce_sum(local_sq);
  __shared__ float smem_sq[kWarpsPerBlock];
  if (lane == 0) {
    smem_sq[warp_id] = warp_sq;
  }
  __syncthreads();
  float total_sq = (lane < kWarpsPerBlock) ? smem_sq[lane] : 0.f;
  if (warp_id == 0) {
    total_sq = warp_reduce_sum(total_sq);
    if (lane == 0) {
      smem_sq[0] = total_sq;
    }
  }
  __syncthreads();
  const float inv_rms = rsqrtf(smem_sq[0] / static_cast<float>(cols) + eps);

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

  // RMS_PARALLAX (legacy) and RMS_AUTO both use stripe; parallax kernel removed.
  (void)kernel_tuning_state();

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
