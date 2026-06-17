export type ModelProgressPhase = "download" | "unloading" | "loading" | "ready";

export type ModelProgressState = {
  phase: ModelProgressPhase;
  label: string;
  percent: number;
  etaSeconds: number | null;
  modelName?: string;
  indeterminate?: boolean;
  totalBytes?: number;
};

const DEFAULT_DOWNLOAD_SPEED_BPS = 8 * 1024 * 1024; // 8 MiB/s
const LOAD_THROUGHPUT_BPS = 150 * 1024 * 1024; // ~150 MiB/s mmap + init

/** Format seconds as m:ss or h:mm:ss (e.g. 5:25). */
export function formatEta(seconds: number | null | undefined): string {
  if (seconds == null || seconds < 0) return "—";
  const total = Math.max(0, Math.round(seconds));
  const hrs = Math.floor(total / 3600);
  const mins = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  if (hrs > 0) return `${hrs}:${pad(mins)}:${pad(secs)}`;
  return `${mins}:${pad(secs)}`;
}

export function guessDownloadSpeedBps(): number {
  const conn = (navigator as Navigator & { connection?: { downlink?: number } }).connection;
  if (typeof conn?.downlink === "number" && conn.downlink > 0) {
    return (conn.downlink * 1_000_000) / 8;
  }
  return DEFAULT_DOWNLOAD_SPEED_BPS;
}

export function computeDownloadEta(
  totalBytes: number,
  bytesDone: number,
  speedBps: number,
  fallbackSpeedBps = guessDownloadSpeedBps(),
): number | null {
  if (totalBytes <= 0) return null;
  const remaining = Math.max(0, totalBytes - bytesDone);
  const speed = speedBps > 0 ? speedBps : fallbackSpeedBps;
  return Math.max(0, Math.round(remaining / speed));
}

export function estimateLoadEtaSeconds(sizeBytes: number): number {
  if (sizeBytes <= 0) return 8;
  return Math.max(5, Math.floor(sizeBytes / LOAD_THROUGHPUT_BPS) + 3);
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
  const serverEta = typeof data.eta_seconds === "number" ? data.eta_seconds : null;
  const speed = typeof data.speed_bps === "number" ? data.speed_bps : 0;
  const repoId = typeof data.repo_id === "string" ? data.repo_id : null;

  if (phase === "resolving" && total <= 0) {
    const label =
      typeof data.label === "string" ? data.label : "Resolving Hugging Face repo…";
    return {
      phase: "download",
      label,
      percent: 0,
      etaSeconds: 45,
      indeterminate: false,
    };
  }

  if (phase === "resolving" && total > 0) {
    const label = typeof data.label === "string" ? data.label : "Resolving Hugging Face repo…";
    return {
      phase: "download",
      label,
      percent: 0,
      etaSeconds: computeDownloadEta(total, 0, 0),
      totalBytes: total,
      indeterminate: false,
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

  const etaSeconds =
    serverEta ??
    computeDownloadEta(total, bytes, speed) ??
    (total > 0 ? computeDownloadEta(total, bytes, 0) : null);

  return {
    phase: "download",
    label,
    percent,
    etaSeconds,
    totalBytes: total > 0 ? total : undefined,
    indeterminate: false,
  };
}

function roundPercent(bytes: number, total: number): number {
  if (total <= 0) return 0;
  return Math.min(100, Math.max(0, Math.round((100 * bytes) / total)));
}

export function initialDownloadProgress(repo: string, totalBytes?: number): ModelProgressState {
  const shortName = repo.split("/").pop() || repo;
  const etaSeconds = totalBytes ? computeDownloadEta(totalBytes, 0, 0) : 45;
  return {
    phase: "download",
    label: `Connecting to Hugging Face · ${repo}`,
    percent: 0,
    etaSeconds,
    totalBytes,
    modelName: shortName,
    indeterminate: false,
  };
}

export function progressFromPreloadEvent(data: Record<string, unknown>): ModelProgressState {
  const phase = (data.phase as ModelProgressPhase) || "loading";
  const percent = typeof data.percent === "number" ? data.percent : phase === "loading" ? 15 : 5;
  const sizeBytes = typeof data.size_bytes === "number" ? data.size_bytes : 0;
  const eta =
    typeof data.eta_seconds === "number" ? data.eta_seconds : estimateLoadEtaSeconds(sizeBytes);
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
    totalBytes: sizeBytes > 0 ? sizeBytes : undefined,
    indeterminate: false,
  };
}

export function initialLoadProgress(name: string, sizeBytes: number): ModelProgressState {
  return {
    phase: "loading",
    label: `Loading ${name} into inference engine`,
    percent: 5,
    etaSeconds: estimateLoadEtaSeconds(sizeBytes),
    modelName: name,
    totalBytes: sizeBytes > 0 ? sizeBytes : undefined,
    indeterminate: false,
  };
}
