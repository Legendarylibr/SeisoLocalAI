#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace seiso {

// ---------------------------------------------------------------------------
// Type traits + float widen for stable reductions
// ---------------------------------------------------------------------------

template <typename T>
struct Widen {
  using type = T;
};

template <>
struct Widen<__half> {
  using type = float;
};

template <>
struct Widen<__nv_bfloat16> {
  using type = float;
};

template <typename T>
__device__ __forceinline__ float to_float(T v) {
  return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<__half>(__half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v);

template <>
__device__ __forceinline__ float from_float<float>(float v) {
  return v;
}

template <>
__device__ __forceinline__ __half from_float<__half>(float v) {
  return __float2half_rn(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

// ---------------------------------------------------------------------------
// Warp reductions (xor-shuffle tree — zero shared memory)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, offset);
  }
  return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
  }
  return v;
}

// ---------------------------------------------------------------------------
// Vectorized loads/stores (width in elements of T)
// ---------------------------------------------------------------------------

template <typename T, int WIDTH>
struct Vec {
  T data[WIDTH];
};

template <typename T, int WIDTH>
__device__ __forceinline__ void load_vec(const T* ptr, int idx, int cols, Vec<T, WIDTH>& v) {
#pragma unroll
  for (int i = 0; i < WIDTH; ++i) {
    const int j = idx + i;
    v.data[i] = (j < cols) ? ptr[j] : T(0);
  }
}

template <typename T, int WIDTH>
__device__ __forceinline__ void store_vec(T* ptr, int idx, int cols, const Vec<T, WIDTH>& v) {
#pragma unroll
  for (int i = 0; i < WIDTH; ++i) {
    const int j = idx + i;
    if (j < cols) {
      ptr[j] = v.data[i];
    }
  }
}

template <typename T, int WIDTH>
__device__ __forceinline__ float vec_sum_sq(const Vec<T, WIDTH>& v) {
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < WIDTH; ++i) {
    const float f = to_float(v.data[i]);
    acc += f * f;
  }
  return acc;
}

}  // namespace seiso
