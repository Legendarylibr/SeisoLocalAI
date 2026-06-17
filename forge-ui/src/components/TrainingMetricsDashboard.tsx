import { useEffect, useMemo, useState } from "react";
import { SystemMetrics, TrainingMetricPoint, api } from "@/lib/api";
import { IconClose } from "@/components/Icons";

type Props = {
  jobId: string | null;
  open: boolean;
  onClose: () => void;
  trainingPoints: TrainingMetricPoint[];
  systemPoints: SystemMetrics[];
  status?: string | null;
};

function Sparkline({
  values,
  color,
  height = 56,
}: {
  values: number[];
  color: string;
  height?: number;
}) {
  if (values.length < 2) {
    return <div className="metrics-sparkline-empty">Waiting for data…</div>;
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const width = 280;
  const points = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * width;
      const y = height - ((v - min) / range) * (height - 8) - 4;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg className="metrics-sparkline" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <polyline fill="none" stroke={color} strokeWidth="2" points={points} />
    </svg>
  );
}

function StatCard({
  label,
  value,
  sub,
  values,
  color,
}: {
  label: string;
  value: string;
  sub?: string;
  values: number[];
  color: string;
}) {
  return (
    <div className="metrics-stat-card">
      <div className="metrics-stat-head">
        <span>{label}</span>
        <span className="metrics-stat-value">{value}</span>
      </div>
      <Sparkline values={values} color={color} />
      {sub && <div className="metrics-stat-sub">{sub}</div>}
    </div>
  );
}

function UtilBar({ label, pct, temp, color }: { label: string; pct: number | null; temp?: number | null; color: string }) {
  const v = pct ?? 0;
  return (
    <div className="metrics-util-row">
      <div className="metrics-util-head">
        <span>{label}</span>
        <span>{pct != null ? `${Math.round(pct)}%` : "—"}</span>
      </div>
      <div className="monitor-bar-track">
        <div className="monitor-bar-fill" style={{ width: `${Math.min(100, v)}%`, background: color }} />
      </div>
      {temp != null && <span className="monitor-temp">{temp}°C</span>}
    </div>
  );
}

export function TrainingMetricsDashboard({
  jobId,
  open,
  onClose,
  trainingPoints,
  systemPoints,
  status,
}: Props) {
  const [now, setNow] = useState(Date.now());
  const [hydratedTraining, setHydratedTraining] = useState<TrainingMetricPoint[]>([]);
  const [hydratedSystem, setHydratedSystem] = useState<SystemMetrics[]>([]);

  useEffect(() => {
    if (!open) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [open]);

  useEffect(() => {
    if (!open || !jobId) return;
    let cancelled = false;
    api.getTrainingMetrics(jobId)
      .then((payload) => {
        if (cancelled) return;
        setHydratedTraining(payload.training ?? []);
        setHydratedSystem(payload.system ?? []);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [open, jobId]);

  const mergedTraining = trainingPoints.length ? trainingPoints : hydratedTraining;
  const mergedSystem = systemPoints.length ? systemPoints : hydratedSystem;

  const training = useMemo(
    () => mergedTraining.filter((p) => p.type === "training" || p.type === "eval"),
    [mergedTraining],
  );
  const losses = training.map((p) => p.loss ?? p.eval_loss).filter((v): v is number => v != null);
  const evalLosses = training.map((p) => p.eval_loss).filter((v): v is number => v != null);
  const rewards = training.map((p) => p.reward).filter((v): v is number => v != null);
  const lrs = training.map((p) => p.learning_rate).filter((v): v is number => v != null);

  const latestTraining = [...training].reverse().find((p) => p.loss != null) ?? training.at(-1);
  const latestEval = [...training].reverse().find((p) => p.eval_loss != null);
  const latestSystem = mergedSystem.at(-1);

  const steps = latestTraining?.step ?? training.at(-1)?.step ?? 0;
  const epoch = latestTraining?.epoch ?? training.at(-1)?.epoch;

  if (!open) return null;

  return (
    <div className="metrics-overlay" role="dialog" aria-label="Training metrics dashboard">
      <div className="metrics-backdrop" onClick={onClose} aria-hidden />
      <div className="metrics-panel">
        <header className="metrics-header">
          <div>
            <h2>Live training metrics</h2>
            <p className="metrics-sub">
              {jobId ? (
                <>
                  Job <span className="mono">{jobId.slice(0, 8)}…</span>
                  {status && <span className={`badge badge-${status}`}>{status}</span>}
                </>
              ) : (
                "No active job"
              )}
            </p>
          </div>
          <button type="button" className="metrics-close" onClick={onClose} aria-label="Close">
            <IconClose size={16} />
          </button>
        </header>

        <div className="metrics-kpi-row">
          <div className="metrics-kpi">
            <span className="metrics-kpi-label">Step</span>
            <span className="metrics-kpi-value">{steps}</span>
          </div>
          <div className="metrics-kpi">
            <span className="metrics-kpi-label">Epoch</span>
            <span className="metrics-kpi-value">{epoch != null ? epoch.toFixed(2) : "—"}</span>
          </div>
          <div className="metrics-kpi">
            <span className="metrics-kpi-label">Loss</span>
            <span className="metrics-kpi-value">
              {latestTraining?.loss != null ? latestTraining.loss.toFixed(4) : "—"}
            </span>
          </div>
          <div className="metrics-kpi">
            <span className="metrics-kpi-label">Eval loss</span>
            <span className="metrics-kpi-value">
              {latestEval?.eval_loss != null ? latestEval.eval_loss.toFixed(4) : "—"}
            </span>
          </div>
          <div className="metrics-kpi">
            <span className="metrics-kpi-label">Reward</span>
            <span className="metrics-kpi-value">
              {latestTraining?.reward != null ? latestTraining.reward.toFixed(4) : "—"}
            </span>
          </div>
        </div>

        <div className="metrics-grid">
          <StatCard
            label="Training loss"
            value={latestTraining?.loss != null ? latestTraining.loss.toFixed(4) : "—"}
            sub={`${losses.length} points`}
            values={losses}
            color="#f87171"
          />
          <StatCard
            label="Eval loss"
            value={latestEval?.eval_loss != null ? latestEval.eval_loss.toFixed(4) : "—"}
            sub={`${evalLosses.length} evals`}
            values={evalLosses}
            color="#60a5fa"
          />
          <StatCard
            label="Reward (−loss)"
            value={latestTraining?.reward != null ? latestTraining.reward.toFixed(4) : "—"}
            sub="Higher is better"
            values={rewards}
            color="#34d399"
          />
          <StatCard
            label="Learning rate"
            value={latestTraining?.learning_rate != null ? latestTraining.learning_rate.toExponential(2) : "—"}
            sub={`${lrs.length} samples`}
            values={lrs}
            color="#c084fc"
          />
        </div>

        <section className="metrics-system">
          <h3>Hardware (live)</h3>
          <p className="metrics-privacy">Read locally · never transmitted · updated {new Date(now).toLocaleTimeString()}</p>
          {latestSystem ? (
            <div className="metrics-system-grid">
              <UtilBar label="CPU" pct={latestSystem.cpu_util_pct} temp={latestSystem.cpu_temp_c} color="var(--accent)" />
              <UtilBar label="RAM" pct={latestSystem.ram_used_pct} color="#60a5fa" />
              {latestSystem.gpus.length > 0 ? (
                latestSystem.gpus.map((g, i) => (
                  <UtilBar
                    key={i}
                    label={g.name.length > 22 ? `${g.name.slice(0, 22)}…` : g.name}
                    pct={g.utilization_pct}
                    temp={g.temperature_c}
                    color="#c084fc"
                  />
                ))
              ) : (
                <p className="muted-text">No discrete GPU metrics available.</p>
              )}
            </div>
          ) : (
            <p className="muted-text">Waiting for system metrics…</p>
          )}
        </section>
      </div>
    </div>
  );
}
