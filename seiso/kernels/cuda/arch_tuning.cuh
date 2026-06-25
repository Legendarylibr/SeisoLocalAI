#pragma once

#include <cuda_runtime.h>

namespace seiso {

// Architecture families for launch-parameter selection.
enum class GpuArchFamily : int {
  UNKNOWN = 0,
  AMPERE = 80,   // sm_80, sm_86
  ADA = 89,      // sm_89
  HOPPER = 90,   // sm_90
  BLACKWELL = 100,  // sm_100+
};

inline GpuArchFamily arch_family_from_sm(int major, int minor) {
  const int sm = major * 10 + minor;
  if (sm >= 100) {
    return GpuArchFamily::BLACKWELL;
  }
  if (sm >= 90) {
    return GpuArchFamily::HOPPER;
  }
  if (sm == 89) {
    return GpuArchFamily::ADA;
  }
  if (sm >= 80) {
    return GpuArchFamily::AMPERE;
  }
  return GpuArchFamily::UNKNOWN;
}

inline GpuArchFamily current_arch_family() {
#if __CUDA_ARCH__ >= 1000
  return GpuArchFamily::BLACKWELL;
#elif __CUDA_ARCH__ >= 900
  return GpuArchFamily::HOPPER;
#elif __CUDA_ARCH__ == 890
  return GpuArchFamily::ADA;
#elif __CUDA_ARCH__ >= 800
  return GpuArchFamily::AMPERE;
#else
  return GpuArchFamily::UNKNOWN;
#endif
}

inline bool arch_has_cp_async() {
#if __CUDA_ARCH__ >= 800
  return true;
#else
  return false;
#endif
}

inline bool arch_has_wmma() {
#if __CUDA_ARCH__ >= 700
  return true;
#else
  return false;
#endif
}

inline bool arch_has_wgmma() {
#if __CUDA_ARCH__ >= 900
  return true;
#else
  return false;
#endif
}

// Autotuned launch defaults per architecture family.
struct ArchLaunchDefaults {
  int lora_tile;
  int mlp_tile;
  int qkv_warps;
  bool use_persistent;
  bool use_wmma;
};

inline ArchLaunchDefaults arch_launch_defaults(GpuArchFamily fam, int lora_tile_override) {
  ArchLaunchDefaults d{};
  d.lora_tile = (lora_tile_override > 0) ? lora_tile_override : 256;
  d.mlp_tile = 256;
  d.qkv_warps = 8;
  d.use_persistent = true;
  d.use_wmma = true;

  switch (fam) {
    case GpuArchFamily::BLACKWELL:
      d.lora_tile = (lora_tile_override > 0) ? lora_tile_override : 512;
      d.mlp_tile = 512;
      d.qkv_warps = 8;
      break;
    case GpuArchFamily::HOPPER:
      d.lora_tile = (lora_tile_override > 0) ? lora_tile_override : 384;
      d.mlp_tile = 384;
      d.qkv_warps = 8;
      break;
    case GpuArchFamily::ADA:
      d.lora_tile = (lora_tile_override > 0) ? lora_tile_override : 256;
      d.mlp_tile = 256;
      d.qkv_warps = 8;
      break;
    case GpuArchFamily::AMPERE:
      d.lora_tile = (lora_tile_override > 0) ? lora_tile_override : 256;
      d.mlp_tile = 256;
      d.qkv_warps = 4;
      break;
    default:
      d.use_persistent = false;
      d.use_wmma = false;
      d.lora_tile = (lora_tile_override > 0) ? lora_tile_override : 128;
      d.mlp_tile = 128;
      d.qkv_warps = 4;
      break;
  }
  return d;
}

}  // namespace seiso