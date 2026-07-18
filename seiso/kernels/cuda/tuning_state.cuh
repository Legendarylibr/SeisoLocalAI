#pragma once

namespace seiso {

enum RmsLaunchMode : int {
  RMS_AUTO = 0,
  RMS_STRIPE = 1,
  // Legacy alias: cp.async parallax path was removed; maps to stripe at launch.
  RMS_PARALLAX = 2,
};

enum SwigluVecMode : int {
  SWIGLU_AUTO = 0,
  SWIGLU_VEC4 = 4,
  SWIGLU_VEC8 = 8,
};

struct KernelTuningState {
  RmsLaunchMode rms_mode = RMS_AUTO;
  SwigluVecMode swiglu_vec = SWIGLU_AUTO;
  int lora_tile = 0;  // 0 = auto dispatch
  int arch_sm = 0;    // e.g. 89 for Ada, 90 for Hopper
  int use_cuda_graphs = 0;
  int use_stream_overlap = 1;
};

inline KernelTuningState& kernel_tuning_state() {
  static KernelTuningState state;
  return state;
}

inline void set_kernel_tuning_state(
    int rms_mode,
    int swiglu_vec,
    int lora_tile,
    int arch_sm = 0,
    int use_cuda_graphs = 0,
    int use_stream_overlap = 1) {
  auto& state = kernel_tuning_state();
  state.rms_mode = static_cast<RmsLaunchMode>(rms_mode);
  state.swiglu_vec = static_cast<SwigluVecMode>(swiglu_vec);
  state.lora_tile = lora_tile;
  state.arch_sm = arch_sm;
  state.use_cuda_graphs = use_cuda_graphs;
  state.use_stream_overlap = use_stream_overlap;
}

}  // namespace seiso
