import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { api, SystemMetrics } from "@/lib/api";

const POLL_MS = 3000;

type MetricsContextValue = {
  metrics: SystemMetrics | null;
  watch: () => () => void;
};

const MetricsContext = createContext<MetricsContextValue | null>(null);

export function MetricsProvider({ children }: { children: ReactNode }) {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [watchers, setWatchers] = useState(0);
  const pollingRef = useRef(false);

  const watch = useCallback(() => {
    setWatchers((count) => count + 1);
    return () => setWatchers((count) => Math.max(0, count - 1));
  }, []);

  useEffect(() => {
    if (watchers <= 0) return;
    let cancelled = false;

    const poll = async () => {
      if (document.hidden) return;
      if (pollingRef.current) return;
      pollingRef.current = true;
      try {
        const next = await api.metrics();
        if (!cancelled) setMetrics(next);
      } catch {
        /* metrics are best-effort and local-only */
      } finally {
        pollingRef.current = false;
      }
    };

    void poll();
    const id = setInterval(poll, POLL_MS);
    document.addEventListener("visibilitychange", poll);
    return () => {
      cancelled = true;
      clearInterval(id);
      document.removeEventListener("visibilitychange", poll);
    };
  }, [watchers]);

  const value = useMemo(() => ({ metrics, watch }), [metrics, watch]);

  return (
    <MetricsContext.Provider value={value}>
      {children}
    </MetricsContext.Provider>
  );
}

export function useLiveMetrics(active = true): SystemMetrics | null {
  const ctx = useContext(MetricsContext);
  if (!ctx) {
    throw new Error("useLiveMetrics must be used within MetricsProvider");
  }

  const { metrics, watch } = ctx;

  useEffect(() => {
    if (!active) return;
    return watch();
  }, [active, watch]);

  return metrics;
}
