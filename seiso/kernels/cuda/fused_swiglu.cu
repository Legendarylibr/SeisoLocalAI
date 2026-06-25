#include "common.cuh"
#include "kernels.cuh"
#include "tuning_state.cuh"

#include <cstdint>
#include <type_traits>

namespace seiso {

// Fused SwiGLU activation: out = silu(gate) * up
// Vectorized, one read pass per tensor, one write.

template <typename T, int VEC, int BLOCK>
__global__ void fused_swiglu_kernel(
    const T* __restrict__ gate,
    const T* __restrict__ up,
    T* __restrict__ out,
    int64_t rows,
    int cols) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  const T* g_row = gate + static_cast<int64_t>(row) * cols;
  const T* u_row = up + static_cast<int64_t>(row) * cols;
  T* o_row = out + static_cast<int64_t>(row) * cols;
  const int vec_cols = cols / VEC;
  const int tid = threadIdx.x;

  for (int i = tid; i < vec_cols; i += BLOCK) {
    const int base = i * VEC;
    if constexpr (std::is_same_v<T, float>) {
      float4 gv = load_vec4(reinterpret_cast<const float*>(g_row) + base);
      float4 uv = load_vec4(reinterpret_cast<const float*>(u_row) + base);
      float4 ov;
      ov.x = silu(gv.x) * uv.x;
      ov.y = silu(gv.y) * uv.y;
      ov.z = silu(gv.z) * uv.z;
      ov.w = silu(gv.w) * uv.w;
      store_vec4(reinterpret_cast<float*>(o_row) + base, ov);
    } else if constexpr (VEC == 8) {
      uint4 gv = load_vec8(reinterpret_cast<const T*>(g_row) + base);
      uint4 uv = load_vec8(reinterpret_cast<const T*>(u_row) + base);
      const T* gh = reinterpret_cast<const T*>(&gv);
      const T* uh = reinterpret_cast<const T*>(&uv);
      T oh[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        float v = silu(to_float(gh[k])) * to_float(uh[k]);
        oh[k] = from_float<T>(v);
      }
      store_vec8(reinterpret_cast<T*>(o_row) + base, *reinterpret_cast<uint4*>(oh));
    } else {
      T gh[4];
      T uh[4];
      T oh[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        gh[k] = g_row[base + k];
        uh[k] = u_row[base + k];
        oh[k] = from_float<T>(silu(to_float(gh[k])) * to_float(uh[k]));
        o_row[base + k] = oh[k];
      }
    }
  }
}

template <typename T>
void launch_fused_swiglu(
    const T* gate,
    const T* up,
    T* out,
    int64_t rows,
    int cols,
    cudaStream_t stream) {
  constexpr int BLOCK = 256;
  const int grid = static_cast<int>(rows);
  const auto& tuning = kernel_tuning_state();
  if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
    const int vec = tuning.swiglu_vec == SWIGLU_VEC4 ? 4 : 8;
    if (vec == 4 && cols % 4 == 0) {
      fused_swiglu_kernel<T, 4, BLOCK><<<grid, BLOCK, 0, stream>>>(gate, up, out, rows, cols);
    } else if (cols % 8 == 0) {
      fused_swiglu_kernel<T, 8, BLOCK><<<grid, BLOCK, 0, stream>>>(gate, up, out, rows, cols);
    } else {
      fused_swiglu_kernel<T, 4, BLOCK><<<grid, BLOCK, 0, stream>>>(gate, up, out, rows, cols);
    }
  } else {
    fused_swiglu_kernel<T, 4, BLOCK><<<grid, BLOCK, 0, stream>>>(gate, up, out, rows, cols);
  }
}

}  // namespace seiso

#include "explicit_inst.cuh"

namespace seiso {

template void launch_fused_swiglu<float>(
    const float*, const float*, float*, int64_t, int, cudaStream_t);
template void launch_fused_swiglu<__half>(
    const __half*, const __half*, __half*, int64_t, int, cudaStream_t);
template void launch_fused_swiglu<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int64_t, int, cudaStream_t);

}  // namespace seiso
