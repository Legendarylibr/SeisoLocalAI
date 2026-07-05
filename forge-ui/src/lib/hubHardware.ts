import type { HardwareSummary, VramStatus } from "@/lib/api";

function headroomGb(hw: HardwareSummary, vram: VramStatus | null): number {
  const mb = vram?.headroom_mb ?? hw.vram_headroom_mb;
  return Math.max(0, Math.round(mb / 1024));
}

function isAppleUnified(hw: HardwareSummary, vram: VramStatus | null): boolean {
  const tier = vram?.tier || hw.tier;
  return Boolean(vram?.apple_unified || tier === "apple_unified");
}

/** Human-readable free memory line for hardware strips. */
export function formatMemoryHeadroom(hw: HardwareSummary, vram: VramStatus | null): string {
  const freeGb = headroomGb(hw, vram);
  const label = (vram?.memory_label || hw.memory_headroom_label || "memory").toUpperCase();
  const ramGb = Math.round(hw.ram_gb || vram?.ram_gb || 0);

  if (isAppleUnified(hw, vram) && ramGb > 0) {
    return `~${freeGb} GB free of ${ramGb} GB unified memory`;
  }
  return `~${freeGb} GB ${label} free`;
}

/** RAM-tier hint for Hub hardware strip (Mac + generic tiers). */
export function hubRamTierHint(hw: HardwareSummary | null, vram: VramStatus | null): string | null {
  if (!hw) return null;
  const ram = hw.ram_gb;
  const tier = vram?.tier || hw.tier;
  const isMac = vram?.apple_unified || tier === "apple_unified" || tier === "cpu_only";
  const freeGb = headroomGb(hw, vram);

  if (isMac && ram > 0) {
    if (freeGb < Math.max(6, ram * 0.3)) {
      return `${Math.round(ram)} GB Mac, but only ~${freeGb} GB free now — close browsers/IDE tabs or other LLM apps, then retry. Model fit uses free memory, not total RAM.`;
    }
    if (ram <= 16) return `${Math.round(ram)} GB Mac — up to ~9B Q4 comfortably (Phi-4 Mini, Gemma 3 4B)`;
    if (ram <= 24) return `${Math.round(ram)} GB Mac — up to ~24B Q4 when enough memory is free`;
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
