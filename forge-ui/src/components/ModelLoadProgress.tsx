import { ModelProgressState, formatEta } from "@/lib/modelProgress";

type Props = {
  progress: ModelProgressState | null;
  modelName?: string | null;
  compact?: boolean;
};

export function ModelLoadProgress({ progress, modelName, compact = false }: Props) {
  if (!progress) return null;

  const pct = Math.min(100, Math.max(0, progress.percent));
  const showIndeterminate = progress.indeterminate && pct < 95;

  return (
    <div className={`model-load-progress${compact ? " model-load-progress-compact" : ""}`}>
      <div className="model-load-progress-header">
        <span className="model-load-progress-label">{progress.label}</span>
        <span className="model-load-progress-eta">
          {showIndeterminate ? "Estimating…" : formatEta(progress.etaSeconds)}
        </span>
      </div>
      <div
        className={`model-load-progress-track${showIndeterminate ? " indeterminate" : ""}`}
        role="progressbar"
        aria-valuenow={showIndeterminate ? undefined : pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={progress.label}
      >
        <div
          className="model-load-progress-fill"
          style={showIndeterminate ? undefined : { width: `${pct}%` }}
        />
      </div>
      {!compact && modelName && progress.phase === "ready" && (
        <p className="model-load-progress-model muted-text">Active: {modelName}</p>
      )}
      {!compact && !showIndeterminate && pct > 0 && (
        <p className="model-load-progress-pct muted-text">{Math.round(pct)}%</p>
      )}
    </div>
  );
}
