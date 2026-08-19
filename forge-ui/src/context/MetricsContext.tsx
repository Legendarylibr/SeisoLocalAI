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

const POLL_MS_ACTIVE = 3000;
const POLL_MS_IDLE = 15000;
const IDLE_AFTER_MS = 30000;

type MetricsContextValue = {
  metrics: SystemMetrics | null;
  watch: () => () => void;
};

const MetricsContext = createContext<MetricsContextValue | null>(null);

export function MetricsProvider({ children }: { children: ReactNode }) {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [watchers, setWatchers] = useState(0);
  const pollingRef = useRef(false);
  const lastChangeRef = useRef(Date.now());
  const lastPayloadRef = useRef<string>("");
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const watch = useCallback(() => {
    setWatchers((count) => count + 1);
    return () => setWatchers((count) => Math.max(0, count - 1));
  }, []);

  useEffect(() => {
    if (watchers <= 0) return;
    let cancelled = false;

    const desiredIntervalMs = () =>
      Date.now() - lastChangeRef.current > IDLE_AFTER_MS
        ? POLL_MS_IDLE
        : POLL_MS_ACTIVE;

    const scheduleNext = () => {
      if (cancelled) return;
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      timeoutRef.current = setTimeout(() => {
        void poll();
      }, desiredIntervalMs());
    };

    const poll = async () => {
      if (cancelled) return;
      if (document.hidden) {
        scheduleNext();
        return;
      }
      if (pollingRef.current) {
        scheduleNext();
        return;
      }
      pollingRef.current = true;
      try {
        const next = await api.metrics();
        if (cancelled) return;
        const fingerprint = JSON.stringify(next);
        if (fingerprint !== lastPayloadRef.current) {
          lastPayloadRef.current = fingerprint;
          lastChangeRef.current = Date.now();
          setMetrics(next);
        }
      } catch {
        /* metrics are best-effort and local-only */
      } finally {
        pollingRef.current = false;
        scheduleNext();
      }
    };

    const onVisibility = () => {
      if (!document.hidden) void poll();
    };

    void poll();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      cancelled = true;
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      document.removeEventListener("visibilitychange", onVisibility);
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
