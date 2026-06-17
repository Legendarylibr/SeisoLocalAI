import { useEffect, useState } from "react";
import { api, SystemMetrics } from "@/lib/api";
import { IconActivity, IconClose } from "@/components/Icons";

export function SystemMonitor() {
  const [open, setOpen] = useState(false);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);

  useEffect(() => {
    if (!open) return;
    const poll = () => api.metrics().then(setMetrics).catch(() => {});
    poll();
    const id = setInterval(poll, 2000);
    return () => clearInterval(id);
  }, [open]);

  const bar = (pct: number | null | undefined, color: string) => {
    const v = pct ?? 0;
    return (
      <div className="monitor-bar-track">
        <div className="monitor-bar-fill" style={{ width: `${Math.min(100, v)}%`, background: color }} />
      </div>
    );
  };

  return (
    <>
      <button
        type="button"
        className="monitor-fab"
        onClick={() => setOpen((v) => !v)}
        title="System monitor (local only)"
        aria-label="Toggle system monitor"
      >
        <span className="monitor-fab-icon">
          <IconActivity size={16} />
        </span>
        {metrics?.gpus[0]?.utilization_pct != null && (
          <span className="monitor-fab-pill">{Math.round(metrics.gpus[0].utilization_pct!)}%</span>
        )}
      </button>

      {open && (
        <div className="monitor-panel" role="dialog" aria-label="System monitor">
          <div className="monitor-header">
            <span>System monitor</span>
            <button type="button" className="monitor-close" onClick={() => setOpen(false)} aria-label="Close">
              <IconClose size={16} />
            </button>
          </div>
          <p className="monitor-privacy">Read locally · never transmitted</p>

          {metrics ? (
            <div className="monitor-body">
              <div className="monitor-row">
                <div className="monitor-row-head">
                  <span>CPU</span>
                  <span>{metrics.cpu_util_pct ?? "—"}%</span>
                </div>
                {bar(metrics.cpu_util_pct, "var(--accent)")}
                {metrics.cpu_temp_c != null && (
                  <span className="monitor-temp">{metrics.cpu_temp_c}°C</span>
                )}
              </div>

              <div className="monitor-row">
                <div className="monitor-row-head">
                  <span>RAM</span>
                  <span>{metrics.ram_used_pct}%</span>
                </div>
                {bar(metrics.ram_used_pct, "#60a5fa")}
              </div>

              {metrics.gpus.map((g, i) => (
                <div key={i} className="monitor-row">
                  <div className="monitor-row-head">
                    <span>{g.name.length > 18 ? g.name.slice(0, 18) + "…" : g.name}</span>
                    <span>{g.utilization_pct ?? "—"}%</span>
                  </div>
                  {bar(g.utilization_pct, "#c084fc")}
                  <div className="monitor-gpu-meta">
                    {g.vram_used_mb != null && g.vram_total_mb != null && (
                      <span>VRAM {Math.round(g.vram_used_mb)}/{Math.round(g.vram_total_mb)} MB</span>
                    )}
                    {g.temperature_c != null && <span>{g.temperature_c}°C</span>}
                  </div>
                </div>
              ))}

              {metrics.gpus.length === 0 && (
                <p className="muted-text" style={{ fontSize: "0.85rem" }}>No discrete GPU metrics available.</p>
              )}
            </div>
          ) : (
            <p className="muted-text" style={{ padding: "1rem" }}>Loading…</p>
          )}
        </div>
      )}
    </>
  );
}
