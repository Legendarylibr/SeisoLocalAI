import type { HardwareProfile, HardwareSummary, VramStatus } from "@/lib/api";

type HeadroomSource = {
  tier?: string;
  effective_vram_mb?: number;
  vram_headroom_mb?: number;
};

/** GPU capacity on discrete cards; live free RAM/VRAM on Apple unified and CPU-only. */
export function uiHeadroomMb(hw: HeadroomSource | null | undefined): number | undefined {
  if (!hw) return undefined;
  if (hw.tier === "apple_unified" || hw.tier === "cpu_only") {
    return hw.vram_headroom_mb;
  }
  return hw.effective_vram_mb ?? hw.vram_headroom_mb;
}

/**
 * Runtime load budget — on discrete GPUs uses the smaller of capacity and live free VRAM
 * so model blocking stays accurate when another model or app already occupies VRAM.
 */
export function loadHeadroomMb(
  hw: HeadroomSource | null | undefined,
  freeVramMb?: number | null,
): number | undefined {
  const capacity = uiHeadroomMb(hw);
  if (capacity == null) return freeVramMb ?? undefined;
  if (isDiscreteGpuPlatform(hw) && freeVramMb != null && freeVramMb > 0) {
    return Math.min(capacity, freeVramMb);
  }
  return capacity;
}

export function uiHeadroomMbFromSummary(
  summary: HardwareSummary | null | undefined,
): number | undefined {
  return uiHeadroomMb(summary);
}

export function uiHeadroomMbFromProfile(
  profile: HardwareProfile | null | undefined,
): number | undefined {
  return uiHeadroomMb(profile);
}

/** Discrete NVIDIA/AMD GPU — not Apple unified or CPU-only. */
export function isDiscreteGpuPlatform(hw: HeadroomSource | null | undefined): boolean {
  if (!hw?.tier) return false;
  return hw.tier !== "apple_unified" && hw.tier !== "cpu_only";
}

/** Native Linux workstation path — discrete GPU VRAM wording. */
export function isNativeLinuxVramPlatform(
  hw: (HeadroomSource & { platform?: string }) | null | undefined,
): boolean {
  if (!isDiscreteGpuPlatform(hw)) return false;
  const platform = hw?.platform?.toLowerCase() ?? "";
  return platform === "linux" || platform === "";
}

export function freeMemoryButtonLabel(hw: HeadroomSource | null | undefined): string {
  return isDiscreteGpuPlatform(hw) ? "Free VRAM" : "Free memory";
}

export function freeMemoryButtonHint(
  hw: (HeadroomSource & { platform?: string }) | null | undefined,
  loadedLabel: string | null | undefined,
): string {
  const action = freeMemoryButtonLabel(hw);
  const loaded = loadedLabel && loadedLabel !== "Nothing loaded" ? loadedLabel : null;
  if (isNativeLinuxVramPlatform(hw)) {
    return loaded
      ? `Unload ${loaded} from GPU VRAM before switching models (${action} keeps your selection).`
      : `${action} — unload the active model from GPU VRAM (keeps your selection).`;
  }
  return loaded
    ? `Unload ${loaded} from RAM/VRAM (${action} keeps your selection).`
    : `${action} — unload the active model from RAM/VRAM (keeps your selection).`;
}

/** RAM-tier hint for Hub hardware strip (Mac + generic tiers). */
export function hubRamTierHint(hw: HardwareSummary | null, vram: VramStatus | null): string | null {
  if (!hw) return null;
  const ram = hw.ram_gb;
  const tier = vram?.tier || hw.tier;
  const isMac = vram?.apple_unified || tier === "apple_unified" || tier === "cpu_only";

  if (isMac && ram > 0) {
    if (ram <= 16) return `${Math.round(ram)} GB Mac — up to ~9B Q4 comfortably (Phi-4 Mini, Gemma 3 4B)`;
    if (ram <= 24) return `${Math.round(ram)} GB Mac — up to ~24B Q4 with free memory first`;
    if (ram <= 32) return `${Math.round(ram)} GB Mac — 27B class models fit with headroom`;
    return `${Math.round(ram)} GB Mac — large models OK; MoE still needs full GGUF in RAM`;
  }

  if (ram > 0 && ram <= 16) return `${Math.round(ram)} GB RAM — prefer ≤7B Q4 for comfortable chat`;
  if (ram > 0 && ram <= 24) return `${Math.round(ram)} GB RAM — up to ~24B Q4 when memory is free`;
  return null;
}

export function formatLoadedModelLabel(vram: VramStatus | null): string {
  if (!vram) return "Nothing loaded";
  const name = vram.active_model || vram.local?.active_model;
  if (!name) return "Nothing loaded";
  return name;
}

export function hasLoadedInferenceMemory(vram: VramStatus | null): boolean {
  if (!vram) return false;
  return Boolean(vram.active_model || vram.local?.active_model);
}
