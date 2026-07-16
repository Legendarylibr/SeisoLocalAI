import type { HardwareSummary, VramStatus } from "@/lib/api";

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
    return `${Math.round(ram)} GB Mac — large models OK; MoE needs full GGUF resident while speed tracks active parameters`;
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
