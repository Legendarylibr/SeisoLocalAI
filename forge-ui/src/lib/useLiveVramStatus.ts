import { useCallback, useEffect, useRef, useState, type Dispatch, type SetStateAction } from "react";
import { api, type VramStatus } from "@/lib/api";

const POLL_MS = 3000;

/** Poll `/models/vram` while mounted — same headroom math, just refreshed. */
export function useLiveVramStatus(active = true): {
  vramStatus: VramStatus | null;
  setVramStatus: Dispatch<SetStateAction<VramStatus | null>>;
  refreshVram: () => Promise<VramStatus | null>;
} {
  const [vramStatus, setVramStatus] = useState<VramStatus | null>(null);
  const pollingRef = useRef(false);

  const refreshVram = useCallback(async () => {
    try {
      const status = await api.vramStatus();
      setVramStatus(status);
      return status;
    } catch (err) {
      console.error(err);
      return null;
    }
  }, []);

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const schedule = () => {
      if (cancelled) return;
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        void poll();
      }, POLL_MS);
    };

    const poll = async () => {
      if (cancelled) return;
      if (document.hidden) {
        schedule();
        return;
      }
      if (pollingRef.current) {
        schedule();
        return;
      }
      pollingRef.current = true;
      try {
        await refreshVram();
      } finally {
        pollingRef.current = false;
        schedule();
      }
    };

    const onVisibility = () => {
      if (!document.hidden) void poll();
    };

    void poll();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [active, refreshVram]);

  return { vramStatus, setVramStatus, refreshVram };
}
