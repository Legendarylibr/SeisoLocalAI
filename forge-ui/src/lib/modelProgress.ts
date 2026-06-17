export type ModelProgressPhase = "download" | "unloading" | "loading" | "ready";

export type ModelProgressState = {
  phase: ModelProgressPhase;
  label: string;
  percent: number;
  etaSeconds: number | null;
  modelName?: string;
  indeterminate?: boolean;
};

export function formatEta(seconds: number | null | undefined): string {
  if (seconds == null || seconds < 0) return "—";
  if (seconds < 5) return "a few seconds";
  if (seconds < 60) return `~${seconds}s`;
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return secs > 0 ? `~${mins}m ${secs}s` : `~${mins}m`;
}

const KIB = 1024;
const MIB = KIB * 1024;
const GIB = MIB * 1024;

export function formatBytes(n: number): string {
  if (n >= GIB) return `${(n / GIB).toFixed(1)} GB`;
  if (n >= MIB) return `${(n / MIB).toFixed(1)} MB`;
  if (n >= KIB) return `${(n / KIB).toFixed(0)} KB`;
  return `${n} B`;
}

export function progressFromDownloadEvent(data: Record<string, unknown>): ModelProgressState {
  const phase = typeof data.phase === "string" ? data.phase : "download";
  const total = typeof data.total_bytes === "number" ? data.total_bytes : 0;
  const bytes = typeof data.bytes === "number" ? data.bytes : 0;
  const eta = typeof data.eta_seconds === "number" ? data.eta_seconds : null;
  const speed = typeof data.speed_bps === "number" ? data.speed_bps : 0;
  const repoId = typeof data.repo_id === "string" ? data.repo_id : null;

  if (phase === "resolving" && total <= 0) {
    const label =
      typeof data.label === "string" ? data.label : "Resolving Hugging Face repo…";
    return {
      phase: "download",
      label,
      percent: 0,
      etaSeconds: null,
      indeterminate: true,
    };
  }

  const percent =
    typeof data.percent === "number"
      ? data.percent
      : total > 0
        ? roundPercent(bytes, total)
        : 0;
  const sizeLabel = total > 0 ? `${formatBytes(bytes)} / ${formatBytes(total)}` : "Downloading…";
  const speedLabel = speed > 0 ? ` · ${formatBytes(speed)}/s` : "";
  const repoLabel = repoId ? ` · ${repoId.split("/").pop()}` : "";
  const label =
    typeof data.label === "string" && data.label
      ? `${data.label}${total > 0 ? ` · ${sizeLabel}${speedLabel}` : ""}`
      : `Downloading model${repoLabel} · ${sizeLabel}${speedLabel}`;

  return {
    phase: "download",
    label,
    percent,
    etaSeconds: eta ?? (total > 0 && speed > 0 ? Math.max(0, Math.round((total - bytes) / speed)) : null),
    indeterminate: total <= 0 && bytes <= 0,
  };
}

function roundPercent(bytes: number, total: number): number {
  if (total <= 0) return 0;
  return Math.min(100, Math.max(0, Math.round((100 * bytes) / total)));
}

export function initialDownloadProgress(repo: string): ModelProgressState {
  const shortName = repo.split("/").pop() || repo;
  return {
    phase: "download",
    label: `Connecting to Hugging Face · ${repo}`,
    percent: 0,
    etaSeconds: null,
    modelName: shortName,
    indeterminate: true,
  };
}

export function progressFromPreloadEvent(data: Record<string, unknown>): ModelProgressState {
  const phase = (data.phase as ModelProgressPhase) || "loading";
  const percent = typeof data.percent === "number" ? data.percent : phase === "loading" ? 20 : 5;
  const eta = typeof data.eta_seconds === "number" ? data.eta_seconds : null;
  const modelName = typeof data.model_name === "string" ? data.model_name : undefined;
  const label =
    typeof data.label === "string"
      ? data.label
      : phase === "unloading"
        ? "Releasing previous model from VRAM"
        : modelName
          ? `Loading ${modelName} into inference engine`
          : "Loading model into inference engine";
  return {
    phase,
    label,
    percent,
    etaSeconds: eta,
    modelName,
    indeterminate: phase === "loading" && percent < 90,
  };
}
