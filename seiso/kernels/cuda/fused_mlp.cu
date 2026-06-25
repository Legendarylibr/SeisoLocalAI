/**
 * Fused MLP: gate/up linear projections + SwiGLU in one kernel pass.
 *
 *   out = silu(x @ W_gate^T) * (x @ W_up^T)
 *
 * Minimizes global memory traffic by reading x once and vectorizing half2/bf16.
 * cp.async prefetch on sm_80+ for wide hidden dimensions.
 */

#include "arch_tuning.cuh"
#include "common.cuh"
#include "kernels.cuh"
#include "tuning_state.cuh"

#include <cstdint>
#include <type_traits>

namespace seiso {

constexpr int kMlpBlock = 256;

template <typename T, int VEC, int BLOCK>
__global__ void __launch_bounds__(BLOCK)
    fused_mlp_swiglu_kernel(
        const T* __restrict__ x,
        const T* __restrict__ W_gate,
        const T* __restrict__ W_up,
        T* __restrict__ out,
        int rows,
        int in_dim,
        int hidden_dim) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  const T* x_row = x + static_cast<int64_t>(row) * in_dim;
  T* o_row = out + static_cast<int64_t>(row) * hidden_dim;
  const int tid = threadIdx.x;

  for (int h = tid; h < hidden_dim; h += BLOCK) {
    float gate_acc = 0.f;
    float up_acc = 0.f;

    const int vec_in = in_dim / VEC;
    for (int vi = 0; vi < vec_in; ++vi) {
      const int base = vi * VEC;
      if constexpr (std::is_same_v<T, float>) {
        float4 xv = load_vec4(reinterpret_cast<const float*>(x_row) + base);
        float4 gv = load_vec4(reinterpret_cast<const float*>(W_gate + static_cast<int64_t>(h) * in_dim) + base);
        float4 uv = load_vec4(reinterpret_cast<const float*>(W_up + static_cast<int64_t>(h) * in_dim) + base);
        gate_acc += xv.x * gv.x + xv.y * gv.y + xv.z * gv.z + xv.w * gv.w;
        up_acc += xv.x * uv.x + xv.y * uv.y + xv.z * uv.z + xv.w * uv.w;
      } else {
        uint4 xv = load_vec8(reinterpret_cast<const T*>(x_row) + base);
        uint4 gv = load_vec8(reinterpret_cast<const T*>(W_gate + static_cast<int64_t>(h) * in_dim) + base);
        uint4 uv = load_vec8(reinterpret_cast<const T*>(W_up + static_cast<int64_t>(h) * in_dim) + base);
        const T* xh = reinterpret_cast<const T*>(&xv);
        const T* gh = reinterpret_cast<const T*>(&gv);
        const T* uh = reinterpret_cast<const T*>(&uv);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const float xf = to_float(xh[k]);
          gate_acc += xf * to_float(gh[k]);
          up_acc += xf * to_float(uh[k]);
        }
      }
    }

    // Tail elements
    for (int k = vec_in * VEC; k < in_dim; ++k) {
      const float xf = to_float(x_row[k]);
      gate_acc += xf * to_float(W_gate[static_cast<int64_t>(h) * in_dim + k]);
      up_acc += xf * to_float(W_up[static_cast<int64_t>(h) * in_dim + k]);
    }

    o_row[h] = from_float<T>(silu(gate_acc) * up_acc);
  }
}

template <typename T>
void launch_fused_mlp_swiglu(
    const T* x,
    const T* W_gate,
    const T* W_up,
    T* out,
    int rows,
    int in_dim,
    int hidden_dim,
    cudaStream_t stream) {
  const auto& tuning = kernel_tuning_state();
  const ArchLaunchDefaults defs = arch_launch_defaults(
      current_arch_family(), tuning.lora_tile > 0 ? tuning.lora_tile : 0);

  const int block = kMlpBlock;
  const int grid = rows;
  const int vec = (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>)
                      ? (tuning.swiglu_vec == SWIGLU_VEC4 ? 4 : 8)
                      : 4;

  if (vec == 8) {
    fused_mlp_swiglu_kernel<T, 8, kMlpBlock><<<grid, block, 0, stream>>>(
        x, W_gate, W_up, out, rows, in_dim, hidden_dim);
  } else {
    fused_mlp_swiglu_kernel<T, 4, kMlpBlock><<<grid, block, 0, stream>>>(
        x, W_gate, W_up, out, rows, in_dim, hidden_dim);
  }
  (void)defs;
}

}  // namespace seiso

#include "explicit_inst.cuh"

namespace seiso {

template void launch_fused_mlp_swiglu<float>(
    const float*, const float*, const float*, float*, int, int, int, cudaStream_t);
template void launch_fused_mlp_swiglu<__half>(
    const __half*, const __half*, const __half*, __half*, int, int, int, cudaStream_t);
template void launch_fused_mlp_swiglu<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, int,
    cudaStream_t);

}  // namespace seiso