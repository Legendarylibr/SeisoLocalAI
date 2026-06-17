import { useEffect, useMemo, useRef, useState } from "react";
import {
  ModelProgressState,
  formatDownloadProgress,
  formatEta,
  formatSpeedBps,
} from "@/lib/modelProgress";

type Props = {
  progress: ModelProgressState | null;
  modelName?: string | null;
};

const STALL_SECONDS = 8;

function useLiveProgress(progress: ModelProgressState) {
  const isDownload = progress.phase === "download";
  const [, setTick] = useState(0);
  const etaAnchor = useRef({ eta: progress.etaSeconds ?? 0, at: Date.now() });
  const creepAnchor = useRef({
    percent: progress.percent,
    at: Date.now(),
    duration: Math.max(progress.etaSeconds ?? 60, 1),
  });
  const lastBytesAt = useRef(Date.now());
  const prevBytesDone = useRef(progress.bytesDone ?? 0);
  const progressKey = `${progress.phase}:${progress.label}:${progress.totalBytes ?? 0}`;
  const prevProgressKey = useRef(progressKey);

  if (progressKey !== prevProgressKey.current) {
    prevProgressKey.current = progressKey;
    lastBytesAt.current = Date.now();
    prevBytesDone.current = progress.bytesDone ?? 0;
  }

  if ((progress.bytesDone ?? 0) > prevBytesDone.current) {
    lastBytesAt.current = Date.now();
    prevBytesDone.current = progress.bytesDone ?? 0;
  }

  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (progress.etaSeconds != null) {
      etaAnchor.current = { eta: progress.etaSeconds, at: Date.now() };
    }
  }, [progress.etaSeconds]);

  useEffect(() => {
    if (progress.percent >= creepAnchor.current.percent) {
      creepAnchor.current.percent = progress.percent;
    }
    if (progress.etaSeconds != null && progress.percent <= 5) {
      creepAnchor.current = {
        percent: progress.percent,
        at: Date.now(),
        duration: Math.max(progress.etaSeconds, 1),
      };
    }
  }, [progress.percent, progress.etaSeconds]);

  const now = Date.now();
  const elapsed = (now - creepAnchor.current.at) / 1000;
  const etaElapsed = (now - etaAnchor.current.at) / 1000;

  const hasByteTracker =
    isDownload && progress.totalBytes != null && progress.totalBytes > 0;

  const hasLiveDownload = isDownload && (progress.bytesDone ?? 0) > 0;
  const liveEta = progress.etaSeconds != null
    ? hasLiveDownload
      ? progress.etaSeconds
      : Math.max(0, Math.round(etaAnchor.current.eta - etaElapsed))
    : null;

  const timePct =
    !isDownload && progress.etaSeconds != null && progress.percent < 98
      ? Math.min(
          98,
          creepAnchor.current.percent +
            (elapsed / creepAnchor.current.duration) *
              Math.max(0, 98 - creepAnchor.current.percent),
        )
      : progress.percent;

  const displayPct = isDownload
    ? progress.percent
    : Math.min(100, Math.max(progress.percent, timePct));

  const stalled =
    hasByteTracker &&
    progress.percent < 100 &&
    now - lastBytesAt.current >= STALL_SECONDS * 1000;

  return { liveEta, displayPct, hasByteTracker, stalled };
}

export function ModelLoadProgress({ progress, modelName }: Props) {
  if (!progress) return null;
  return (
    <ModelLoadProgressView progress={progress} modelName={modelName} />
  );
}

type ViewProps = {
  progress: ModelProgressState;
  modelName?: string | null;
};

function ModelLoadProgressView({ progress, modelName }: ViewProps) {
  const { liveEta, displayPct, hasByteTracker, stalled } = useLiveProgress(progress);
  const pct = Math.min(100, Math.max(0, displayPct));
  const bytesDone = progress.bytesDone ?? 0;
  const totalBytes = progress.totalBytes ?? 0;
  const speedLabel = formatSpeedBps(progress.speedBps ?? 0);
  const progressLabel = useMemo(
    () => (hasByteTracker ? formatDownloadProgress(bytesDone, totalBytes) : ""),
    [hasByteTracker, bytesDone, totalBytes],
  );

  return (
    <div
      className={`model-load-progress${hasByteTracker ? " model-load-progress-download" : ""}`}
    >
      <div className="model-load-progress-header">
        <span className="model-load-progress-label">{progress.label}</span>
        <span className="model-load-progress-eta">{formatEta(liveEta)}</span>
      </div>
      {hasByteTracker && (
        <div className="model-load-progress-bytes">
          <span className="model-load-progress-bytes-count">{progressLabel}</span>
          {speedLabel ? (
            <span className="model-load-progress-bytes-speed">{speedLabel}</span>
          ) : stalled ? (
            <span className="model-load-progress-bytes-stalled">
              {bytesDone > 0 ? "Waiting for more data…" : "Waiting for first data…"}
            </span>
          ) : null}
        </div>
      )}
      <div
        className="model-load-progress-track"
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={progress.label}
      >
        <div className="model-load-progress-fill" style={{ width: `${pct}%` }} />
      </div>
      {modelName && progress.phase === "ready" && (
        <p className="model-load-progress-model muted-text">Active: {modelName}</p>
      )}
      {pct > 0 && (
        <p className="model-load-progress-pct muted-text">{pct.toFixed(hasByteTracker ? 1 : 0)}%</p>
      )}
    </div>
  );
}
