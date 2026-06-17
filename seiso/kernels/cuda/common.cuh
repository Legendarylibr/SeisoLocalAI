#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace seiso {

// ---------------------------------------------------------------------------
// Type traits
// ---------------------------------------------------------------------------

template <typename T>
struct VecWidth;

template <>
struct VecWidth<float> {
  static constexpr int k = 4;
  using type = float4;
};

template <>
struct VecWidth<__half> {
  static constexpr int k = 8;
  using type = uint4;
};

template <>
struct VecWidth<__nv_bfloat16> {
  static constexpr int k = 8;
  using type = uint4;
};

// ---------------------------------------------------------------------------
// Device math
// ---------------------------------------------------------------------------

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ float from_float<float>(float x) {
  return x;
}

template <>
__device__ __forceinline__ __half from_float<__half>(float x) {
  return __float2half(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}

__device__ __forceinline__ float to_float(float4 v) { return v.x; }
__device__ __forceinline__ float4 from_float4(float x) { return make_float4(x, x, x, x); }

__device__ __forceinline__ float4 load_vec4(const float* ptr) {
  return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ uint4 load_vec8(const __half* ptr) {
  return *reinterpret_cast<const uint4*>(ptr);
}

__device__ __forceinline__ uint4 load_vec8(const __nv_bfloat16* ptr) {
  return *reinterpret_cast<const uint4*>(ptr);
}

__device__ __forceinline__ void store_vec4(float* ptr, float4 v) {
  *reinterpret_cast<float4*>(ptr) = v;
}

__device__ __forceinline__ void store_vec8(__half* ptr, uint4 v) {
  *reinterpret_cast<uint4*>(ptr) = v;
}

__device__ __forceinline__ void store_vec8(__nv_bfloat16* ptr, uint4 v) {
  *reinterpret_cast<uint4*>(ptr) = v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__device__ __forceinline__ float block_sum(float v, float* smem) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) {
    smem[wid] = v;
  }
  __syncthreads();
  v = (threadIdx.x < (blockDim.x + 31) / 32) ? smem[threadIdx.x] : 0.f;
  if (wid == 0) {
    v = warp_sum(v);
  }
  return v;
}

__device__ __forceinline__ float silu(float x) {
  return x / (1.f + expf(-x));
}

// Ampere+ async copy helpers (no-op fallback on older arch at compile time)
#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp_async_ca(const void* src, void* dst, int size) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(size));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
#endif

}  // namespace seiso
