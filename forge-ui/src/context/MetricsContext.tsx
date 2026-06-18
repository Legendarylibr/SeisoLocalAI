import {
  createContext,
  useCallback,
  useContext,
  useEffect,
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

  const watch = useCallback(() => {
    setWatchers((count) => count + 1);
    return () => setWatchers((count) => Math.max(0, count - 1));
  }, []);

  useEffect(() => {
    if (watchers <= 0) return;

    const poll = () => {
      if (document.hidden) return;
      api.metrics().then(setMetrics).catch(() => {});
    };

    poll();
    const id = setInterval(poll, POLL_MS);
    document.addEventListener("visibilitychange", poll);
    return () => {
      clearInterval(id);
      document.removeEventListener("visibilitychange", poll);
    };
  }, [watchers]);

  return (
    <MetricsContext.Provider value={{ metrics, watch }}>
      {children}
    </MetricsContext.Provider>
  );
}

export function useLiveMetrics(active = true): SystemMetrics | null {
  const ctx = useContext(MetricsContext);
  if (!ctx) {
    throw new Error("useLiveMetrics must be used within MetricsProvider");
  }

  useEffect(() => {
    if (!active) return;
    return ctx.watch();
  }, [active, ctx]);

  return ctx.metrics;
}
