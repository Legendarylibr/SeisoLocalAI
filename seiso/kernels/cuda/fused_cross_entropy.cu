/**
 * Fused cross-entropy forward/backward without materializing full softmax.
 *
 * P2 polish:
 *   - Vectorized logit loads (float4 / 8× half-bf16 via uint4)
 *   - Warp-shuffle block reduce (2 __syncthreads vs tree of 8)
 */

#include "include/seiso_vec.cuh"

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace seiso {

constexpr int kCeBlock = 256;
constexpr int kCeWarps = kCeBlock / 32;  // 8

// ---------------------------------------------------------------------------
// Block reductions via warp shuffles (two barriers only)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float block_reduce_max(float val) {
  __shared__ float smem[kCeWarps];
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;

  val = warp_reduce_max(val);
  if (lane == 0) {
    smem[wid] = val;
  }
  __syncthreads();

  val = (threadIdx.x < kCeWarps) ? smem[threadIdx.x] : -INFINITY;
  if (wid == 0) {
    val = warp_reduce_max(val);
  }
  if (lane == 0 && wid == 0) {
    smem[0] = val;
  }
  __syncthreads();
  return smem[0];
}

__device__ __forceinline__ float block_reduce_sum(float val) {
  __shared__ float smem[kCeWarps];
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;

  val = warp_reduce_sum(val);
  if (lane == 0) {
    smem[wid] = val;
  }
  __syncthreads();

  val = (threadIdx.x < kCeWarps) ? smem[threadIdx.x] : 0.f;
  if (wid == 0) {
    val = warp_reduce_sum(val);
  }
  if (lane == 0 && wid == 0) {
    smem[0] = val;
  }
  __syncthreads();
  return smem[0];
}

// ---------------------------------------------------------------------------
// Vectorized pass helpers
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ void ce_accum_max_sum(
    const T* __restrict__ row_logits,
    int vocab,
    float max_logit,
    float& thread_max,
    float& thread_sum,
    bool accumulate_sum) {
  constexpr int kVec =
      std::is_same_v<T, float> ? 4 : 8;
  const int vec_end = (vocab / kVec) * kVec;
  const int tid = static_cast<int>(threadIdx.x);

  if constexpr (std::is_same_v<T, float>) {
    for (int base = tid * kVec; base < vec_end; base += kCeBlock * kVec) {
      const float4 v = *reinterpret_cast<const float4*>(row_logits + base);
      if (!accumulate_sum) {
        thread_max = fmaxf(thread_max, v.x);
        thread_max = fmaxf(thread_max, v.y);
        thread_max = fmaxf(thread_max, v.z);
        thread_max = fmaxf(thread_max, v.w);
      } else {
        thread_sum += expf(v.x - max_logit);
        thread_sum += expf(v.y - max_logit);
        thread_sum += expf(v.z - max_logit);
        thread_sum += expf(v.w - max_logit);
      }
    }
  } else {
    for (int base = tid * kVec; base < vec_end; base += kCeBlock * kVec) {
      const uint4 packed = *reinterpret_cast<const uint4*>(row_logits + base);
      const T* elems = reinterpret_cast<const T*>(&packed);
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        const float f = to_float(elems[k]);
        if (!accumulate_sum) {
          thread_max = fmaxf(thread_max, f);
        } else {
          thread_sum += expf(f - max_logit);
        }
      }
    }
  }

  for (int i = vec_end + tid; i < vocab; i += kCeBlock) {
    const float f = to_float(row_logits[i]);
    if (!accumulate_sum) {
      thread_max = fmaxf(thread_max, f);
    } else {
      thread_sum += expf(f - max_logit);
    }
  }
}

template <typename T>
__device__ __forceinline__ void ce_write_grad(
    const T* __restrict__ row_logits,
    T* __restrict__ row_grad,
    int vocab,
    int label,
    float max_logit,
    float lse,
    float inv_count) {
  constexpr int kVec = std::is_same_v<T, float> ? 4 : 8;
  const int vec_end = (vocab / kVec) * kVec;
  const int tid = static_cast<int>(threadIdx.x);
  const float inv_denom = expf(-(lse - max_logit));  // 1 / sum(exp(x-max))

  if constexpr (std::is_same_v<T, float>) {
    for (int base = tid * kVec; base < vec_end; base += kCeBlock * kVec) {
      const float4 v = *reinterpret_cast<const float4*>(row_logits + base);
      float4 g;
      g.x = expf(v.x - max_logit) * inv_denom * inv_count;
      g.y = expf(v.y - max_logit) * inv_denom * inv_count;
      g.z = expf(v.z - max_logit) * inv_denom * inv_count;
      g.w = expf(v.w - max_logit) * inv_denom * inv_count;
      if (label >= base && label < base + 4) {
        float* gp = reinterpret_cast<float*>(&g);
        gp[label - base] -= inv_count;
      }
      *reinterpret_cast<float4*>(row_grad + base) = g;
    }
  } else {
    for (int base = tid * kVec; base < vec_end; base += kCeBlock * kVec) {
      const uint4 packed = *reinterpret_cast<const uint4*>(row_logits + base);
      const T* elems = reinterpret_cast<const T*>(&packed);
      T out_elems[kVec];
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        float prob = expf(to_float(elems[k]) - max_logit) * inv_denom;
        if (base + k == label) {
          prob -= 1.f;
        }
        out_elems[k] = from_float<T>(prob * inv_count);
      }
      *reinterpret_cast<uint4*>(row_grad + base) = *reinterpret_cast<uint4*>(out_elems);
    }
  }

  for (int i = vec_end + tid; i < vocab; i += kCeBlock) {
    float prob = expf(to_float(row_logits[i]) - max_logit) * inv_denom;
    if (i == label) {
      prob -= 1.f;
    }
    row_grad[i] = from_float<T>(prob * inv_count);
  }
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

template <typename T>
__global__ void __launch_bounds__(kCeBlock)
    cross_entropy_forward_kernel(
        const T* __restrict__ logits,
        const int64_t* __restrict__ labels,
        float* __restrict__ row_loss,
        float* __restrict__ row_max,
        float* __restrict__ row_lse,
        int vocab,
        int ignore_index) {
  const int row = static_cast<int>(blockIdx.x);
  const int label = static_cast<int>(labels[row]);
  const T* row_logits = logits + static_cast<int64_t>(row) * vocab;

  if (label == ignore_index) {
    if (threadIdx.x == 0) {
      row_loss[row] = 0.f;
      row_max[row] = 0.f;
      row_lse[row] = 1.f;
    }
    return;
  }

  float thread_max = -INFINITY;
  float dummy_sum = 0.f;
  ce_accum_max_sum<T>(row_logits, vocab, 0.f, thread_max, dummy_sum, /*accumulate_sum=*/false);
  const float max_logit = block_reduce_max(thread_max);

  float thread_sum = 0.f;
  float unused_max = 0.f;
  ce_accum_max_sum<T>(row_logits, vocab, max_logit, unused_max, thread_sum, /*accumulate_sum=*/true);
  const float block_sum = block_reduce_sum(thread_sum);
  const float lse = logf(fmaxf(block_sum, 1e-20f)) + max_logit;

  if (threadIdx.x == 0) {
    const float target = to_float(row_logits[label]);
    row_loss[row] = lse - target;
    row_max[row] = max_logit;
    row_lse[row] = lse;
  }
}

template <typename T>
__global__ void __launch_bounds__(kCeBlock)
    cross_entropy_backward_kernel(
        const T* __restrict__ logits,
        const int64_t* __restrict__ labels,
        const float* __restrict__ row_max,
        const float* __restrict__ row_lse,
        T* __restrict__ grad_logits,
        int vocab,
        int ignore_index,
        float inv_count) {
  const int row = static_cast<int>(blockIdx.x);
  const int label = static_cast<int>(labels[row]);
  const T* row_logits = logits + static_cast<int64_t>(row) * vocab;
  T* row_grad = grad_logits + static_cast<int64_t>(row) * vocab;

  if (label == ignore_index) {
    constexpr int kVec = std::is_same_v<T, float> ? 4 : 8;
    const int vec_end = (vocab / kVec) * kVec;
    const int tid = static_cast<int>(threadIdx.x);
    if constexpr (std::is_same_v<T, float>) {
      const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int base = tid * kVec; base < vec_end; base += kCeBlock * kVec) {
        *reinterpret_cast<float4*>(row_grad + base) = z;
      }
    } else {
      const uint4 z = make_uint4(0, 0, 0, 0);
      for (int base = tid * kVec; base < vec_end; base += kCeBlock * kVec) {
        *reinterpret_cast<uint4*>(row_grad + base) = z;
      }
    }
    for (int i = vec_end + tid; i < vocab; i += kCeBlock) {
      row_grad[i] = from_float<T>(0.f);
    }
    return;
  }

  ce_write_grad<T>(
      row_logits,
      row_grad,
      vocab,
      label,
      row_max[row],
      row_lse[row],
      inv_count);
}

template <typename T>
void launch_cross_entropy_forward(
    const T* logits,
    const int64_t* labels,
    float* row_loss,
    float* row_max,
    float* row_lse,
    int rows,
    int vocab,
    int ignore_index,
    cudaStream_t stream) {
  cross_entropy_forward_kernel<T><<<rows, kCeBlock, 0, stream>>>(
      logits, labels, row_loss, row_max, row_lse, vocab, ignore_index);
}

template <typename T>
void launch_cross_entropy_backward(
    const T* logits,
    const int64_t* labels,
    const float* row_max,
    const float* row_lse,
    T* grad_logits,
    int rows,
    int vocab,
    int ignore_index,
    float inv_count,
    cudaStream_t stream) {
  cross_entropy_backward_kernel<T><<<rows, kCeBlock, 0, stream>>>(
      logits, labels, row_max, row_lse, grad_logits, vocab, ignore_index, inv_count);
}

}  // namespace seiso

#include "explicit_inst.cuh"

namespace seiso {

template void launch_cross_entropy_forward<float>(
    const float*, const int64_t*, float*, float*, float*, int, int, int, cudaStream_t);
template void launch_cross_entropy_forward<__half>(
    const __half*, const int64_t*, float*, float*, float*, int, int, int, cudaStream_t);
template void launch_cross_entropy_forward<__nv_bfloat16>(
    const __nv_bfloat16*, const int64_t*, float*, float*, float*, int, int, int, cudaStream_t);

template void launch_cross_entropy_backward<float>(
    const float*, const int64_t*, const float*, const float*, float*, int, int, int, float,
    cudaStream_t);
template void launch_cross_entropy_backward<__half>(
    const __half*, const int64_t*, const float*, const float*, __half*, int, int, int, float,
    cudaStream_t);
template void launch_cross_entropy_backward<__nv_bfloat16>(
    const __nv_bfloat16*, const int64_t*, const float*, const float*, __nv_bfloat16*, int, int, int,
    float, cudaStream_t);

}  // namespace seiso
