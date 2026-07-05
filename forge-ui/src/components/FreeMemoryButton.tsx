import { IconEject } from "@/components/Icons";
import {
  freeMemoryButtonHint,
  freeMemoryButtonLabel,
  formatLoadedModelLabel,
} from "@/lib/hubHardware";
import type { HardwareProfile, HardwareSummary, VramStatus } from "@/lib/api";

type FreeMemoryHw = HardwareSummary | HardwareProfile | VramStatus | null | undefined;

type FreeMemoryButtonProps = {
  hw?: FreeMemoryHw;
  vram?: VramStatus | null;
  loading?: boolean;
  disabled?: boolean;
  onClick: () => void;
  /** hub = full label in hardware strip; chat = compact in model bar */
  variant?: "hub" | "chat";
};

export function FreeMemoryButton({
  hw,
  vram,
  loading = false,
  disabled = false,
  onClick,
  variant = "hub",
}: FreeMemoryButtonProps) {
  const loadedLabel = formatLoadedModelLabel(vram ?? null);
  const label = freeMemoryButtonLabel(hw ?? vram);
  const hint = freeMemoryButtonHint(hw ?? vram, loadedLabel);
  const vramMode = label === "Free VRAM";

  return (
    <button
      type="button"
      className={`free-memory-btn${vramMode ? " free-memory-btn--vram" : ""}${variant === "chat" ? " free-memory-btn--chat" : ""}`}
      onClick={onClick}
      disabled={disabled || loading}
      title={hint}
      aria-label={loading ? `Releasing ${label.toLowerCase()}` : label}
    >
      <IconEject size={variant === "chat" ? 15 : 16} />
      <span className="free-memory-btn-label">{loading ? "Freeing…" : label}</span>
      {variant === "hub" && loadedLabel !== "Nothing loaded" && (
        <span className="free-memory-btn-meta">{loadedLabel}</span>
      )}
    </button>
  );
}