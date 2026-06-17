#include "common.cuh"
#include "kernels.cuh"

#include <cstdint>
#include <type_traits>

namespace seiso {

// Fused: out = rms_norm(x + residual) * weight
// Single global read pass for x/residual/weight, single write for out.
// Two-phase within CTA: (1) variance reduction, (2) normalize + scale.

template <typename T, int VEC, int BLOCK>
__global__ void fused_residual_rmsnorm_kernel(
    const T* __restrict__ x,
    const T* __restrict__ residual,
    const T* __restrict__ weight,
    T* __restrict__ out,
    int64_t rows,
    int cols,
    float eps) {
  extern __shared__ float smem_reduce[];
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  const T* x_row = x + static_cast<int64_t>(row) * cols;
  const T* r_row = residual ? (residual + static_cast<int64_t>(row) * cols) : nullptr;
  T* o_row = out + static_cast<int64_t>(row) * cols;

  float sum_sq = 0.f;
  const int vec_cols = cols / VEC;
  const int tid = threadIdx.x;

  for (int i = tid; i < vec_cols; i += BLOCK) {
    const int base = i * VEC;
    if constexpr (std::is_same_v<T, float>) {
      float4 xv = load_vec4(reinterpret_cast<const float*>(x_row) + base);
      float4 rv = r_row ? load_vec4(reinterpret_cast<const float*>(r_row) + base)
                        : make_float4(0.f, 0.f, 0.f, 0.f);
      sum_sq += (xv.x + rv.x) * (xv.x + rv.x);
      sum_sq += (xv.y + rv.y) * (xv.y + rv.y);
      sum_sq += (xv.z + rv.z) * (xv.z + rv.z);
      sum_sq += (xv.w + rv.w) * (xv.w + rv.w);
    } else {
      uint4 xv = load_vec8(reinterpret_cast<const T*>(x_row) + base);
      uint4 rv = r_row ? load_vec8(reinterpret_cast<const T*>(r_row) + base) : make_uint4(0, 0, 0, 0);
      const T* xh = reinterpret_cast<const T*>(&xv);
      const T* rh = reinterpret_cast<const T*>(&rv);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        float v = to_float(xh[k]) + (r_row ? to_float(rh[k]) : 0.f);
        sum_sq += v * v;
      }
    }
  }

  sum_sq = block_sum(sum_sq, smem_reduce);
  __shared__ float inv_rms;
  if (tid == 0) {
    inv_rms = rsqrtf(sum_sq / static_cast<float>(cols) + eps);
  }
  __syncthreads();

  for (int i = tid; i < vec_cols; i += BLOCK) {
    const int base = i * VEC;
    if constexpr (std::is_same_v<T, float>) {
      float4 xv = load_vec4(reinterpret_cast<const float*>(x_row) + base);
      float4 rv = r_row ? load_vec4(reinterpret_cast<const float*>(r_row) + base)
                        : make_float4(0.f, 0.f, 0.f, 0.f);
      float4 wv = load_vec4(reinterpret_cast<const float*>(weight) + base);
      float4 ov;
      ov.x = (xv.x + rv.x) * inv_rms * wv.x;
      ov.y = (xv.y + rv.y) * inv_rms * wv.y;
      ov.z = (xv.z + rv.z) * inv_rms * wv.z;
      ov.w = (xv.w + rv.w) * inv_rms * wv.w;
      store_vec4(reinterpret_cast<float*>(o_row) + base, ov);
    } else {
      uint4 xv = load_vec8(reinterpret_cast<const T*>(x_row) + base);
      uint4 rv = r_row ? load_vec8(reinterpret_cast<const T*>(r_row) + base) : make_uint4(0, 0, 0, 0);
      uint4 wv = load_vec8(reinterpret_cast<const T*>(weight) + base);
      const T* xh = reinterpret_cast<const T*>(&xv);
      const T* rh = reinterpret_cast<const T*>(&rv);
      const T* wh = reinterpret_cast<const T*>(&wv);
      T oh[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        float v = (to_float(xh[k]) + (r_row ? to_float(rh[k]) : 0.f)) * inv_rms * to_float(wh[k]);
        oh[k] = from_float<T>(v);
      }
      store_vec8(reinterpret_cast<T*>(o_row) + base, *reinterpret_cast<uint4*>(oh));
    }
  }
}

// Plain RMSNorm without residual — same kernel, nullptr residual.
template <typename T, int VEC, int BLOCK>
__global__ void fused_rmsnorm_kernel(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ out,
    int64_t rows,
    int cols,
    float eps) {
  fused_residual_rmsnorm_kernel<T, VEC, BLOCK>(x, nullptr, weight, out, rows, cols, eps);
}

template <typename T>
void launch_fused_rmsnorm(
    const T* x,
    const T* residual,
    const T* weight,
    T* out,
    int64_t rows,
    int cols,
    float eps,
    cudaStream_t stream) {
  constexpr int BLOCK = 256;
  int vec = 4;
  if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
    vec = 8;
  }
  const int grid = static_cast<int>(rows);
  const int smem = ((BLOCK + 31) / 32) * sizeof(float);

  if (vec == 8) {
    fused_residual_rmsnorm_kernel<T, 8, BLOCK>
        <<<grid, BLOCK, smem, stream>>>(x, residual, weight, out, rows, cols, eps);
  } else {
    fused_residual_rmsnorm_kernel<T, 4, BLOCK>
        <<<grid, BLOCK, smem, stream>>>(x, residual, weight, out, rows, cols, eps);
  }
}

}  // namespace seiso
