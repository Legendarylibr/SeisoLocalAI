#include "include/seiso_vec.cuh"

#include <cmath>
#include <cstdint>

namespace seiso {

constexpr int kCeBlock = 256;

// Per-row forward: stable log-softmax loss without materializing full softmax.
template <typename T>
__global__ void cross_entropy_forward_kernel(
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
    row_loss[row] = 0.f;
    row_max[row] = 0.f;
    row_lse[row] = 1.f;
    return;
  }

  float thread_max = -INFINITY;
  for (int i = threadIdx.x; i < vocab; i += kCeBlock) {
    thread_max = fmaxf(thread_max, to_float(row_logits[i]));
  }

  __shared__ float smem[kCeBlock];
  smem[threadIdx.x] = thread_max;
  __syncthreads();

  float block_max = thread_max;
  for (int stride = kCeBlock / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      block_max = fmaxf(block_max, smem[threadIdx.x + stride]);
      smem[threadIdx.x] = block_max;
    }
    __syncthreads();
  }
  const float max_logit = smem[0];

  float thread_sum = 0.f;
  for (int i = threadIdx.x; i < vocab; i += kCeBlock) {
    thread_sum += expf(to_float(row_logits[i]) - max_logit);
  }

  smem[threadIdx.x] = thread_sum;
  __syncthreads();

  float block_sum = thread_sum;
  for (int stride = kCeBlock / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      block_sum += smem[threadIdx.x + stride];
      smem[threadIdx.x] = block_sum;
    }
    __syncthreads();
  }
  const float lse = logf(fmaxf(block_sum, 1e-20f)) + max_logit;

  if (threadIdx.x == 0) {
    const float target = to_float(row_logits[label]);
    row_loss[row] = lse - target;
    row_max[row] = max_logit;
    row_lse[row] = lse;
  }
}

template <typename T>
__global__ void cross_entropy_backward_kernel(
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
    for (int i = threadIdx.x; i < vocab; i += kCeBlock) {
      row_grad[i] = from_float<T>(0.f);
    }
    return;
  }

  const float max_logit = row_max[row];
  const float lse = row_lse[row];

  for (int i = threadIdx.x; i < vocab; i += kCeBlock) {
    float prob = expf(to_float(row_logits[i]) - max_logit - (lse - max_logit));
    if (i == label) {
      prob -= 1.f;
    }
    row_grad[i] = from_float<T>(prob * inv_count);
  }
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
