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
import { api, HardwareProfile } from "@/lib/api";
import { useAuth } from "@/hooks/useAuth";

type HardwareProfileState = {
  profile: HardwareProfile | null;
  loading: boolean;
  refresh: () => Promise<void>;
};

const HardwareProfileContext = createContext<HardwareProfileState | null>(null);

export function HardwareProfileProvider({ children }: { children: ReactNode }) {
  const { user, loading: authLoading } = useAuth();
  const [profile, setProfile] = useState<HardwareProfile | null>(null);
  const [loading, setLoading] = useState(false);
  const cacheRef = useRef<{ userId: string; profile: HardwareProfile } | null>(null);

  const refresh = useCallback(async () => {
    if (!user) {
      setProfile(null);
      return;
    }
    setLoading(true);
    try {
      const next = await api.hardware();
      cacheRef.current = { userId: user.id, profile: next };
      setProfile(next);
    } catch {
      /* profile stays local-only; keep last known value on transient errors */
    } finally {
      setLoading(false);
    }
  }, [user]);

  useEffect(() => {
    if (authLoading) return;
    if (!user) {
      setProfile(null);
      cacheRef.current = null;
      setLoading(false);
      return;
    }
    const cached = cacheRef.current;
    if (cached?.userId === user.id) {
      setProfile(cached.profile);
      return;
    }
    void refresh();
  }, [user, authLoading, refresh]);

  const value = useMemo(() => ({ profile, loading, refresh }), [profile, loading, refresh]);

  return <HardwareProfileContext.Provider value={value}>{children}</HardwareProfileContext.Provider>;
}

export function useHardwareProfileContext(): HardwareProfileState {
  const ctx = useContext(HardwareProfileContext);
  if (!ctx) {
    throw new Error("useHardwareProfileContext must be used within HardwareProfileProvider");
  }
  return ctx;
}
