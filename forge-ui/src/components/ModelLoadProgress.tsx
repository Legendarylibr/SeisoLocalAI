import { useEffect, useRef, useState } from "react";
import { ModelProgressState, formatEta } from "@/lib/modelProgress";

type Props = {
  progress: ModelProgressState | null;
  modelName?: string | null;
  compact?: boolean;
};

function useLiveProgress(progress: ModelProgressState) {
  const [now, setNow] = useState(() => Date.now());
  const etaAnchor = useRef({ eta: progress.etaSeconds ?? 0, at: Date.now() });
  const creepAnchor = useRef({
    percent: progress.percent,
    at: Date.now(),
    duration: Math.max(progress.etaSeconds ?? 60, 1),
  });

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
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

  const elapsed = (now - creepAnchor.current.at) / 1000;
  const etaElapsed = (now - etaAnchor.current.at) / 1000;

  const hasLiveDownload = progress.phase === "download" && progress.percent > 0;
  const liveEta = progress.etaSeconds != null
    ? hasLiveDownload
      ? progress.etaSeconds
      : Math.max(0, Math.round(etaAnchor.current.eta - etaElapsed))
    : null;

  const timePct =
    progress.etaSeconds != null && progress.percent < 98
      ? Math.min(
          98,
          creepAnchor.current.percent +
            (elapsed / creepAnchor.current.duration) *
              Math.max(0, 98 - creepAnchor.current.percent),
        )
      : progress.percent;

  const displayPct = Math.min(100, Math.max(progress.percent, timePct));

  return { liveEta, displayPct };
}

export function ModelLoadProgress({ progress, modelName, compact = false }: Props) {
  if (!progress) return null;
  return (
    <ModelLoadProgressView progress={progress} modelName={modelName} compact={compact} />
  );
}

type ViewProps = {
  progress: ModelProgressState;
  modelName?: string | null;
  compact?: boolean;
};

function ModelLoadProgressView({ progress, modelName, compact = false }: ViewProps) {
  const { liveEta, displayPct } = useLiveProgress(progress);
  const pct = Math.min(100, Math.max(0, displayPct));

  return (
    <div className={`model-load-progress${compact ? " model-load-progress-compact" : ""}`}>
      <div className="model-load-progress-header">
        <span className="model-load-progress-label">{progress.label}</span>
        <span className="model-load-progress-eta">{formatEta(liveEta)}</span>
      </div>
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
      {!compact && modelName && progress.phase === "ready" && (
        <p className="model-load-progress-model muted-text">Active: {modelName}</p>
      )}
      {!compact && pct > 0 && (
        <p className="model-load-progress-pct muted-text">{Math.round(pct)}%</p>
      )}
    </div>
  );
}
