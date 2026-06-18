import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { api, HfHubStatus } from "@/lib/api";
import { invalidateApiCache, refreshCachedGet } from "@/lib/api/getCache";
import { useAuth } from "@/hooks/useAuth";

type SettingsResponse = Awaited<ReturnType<typeof api.settings>>;

type PlatformSettingsState = {
  settings: SettingsResponse | null;
  hfStatus: HfHubStatus | null;
  loading: boolean;
  refresh: () => Promise<void>;
};

const PlatformSettingsContext = createContext<PlatformSettingsState | null>(null);

export function PlatformSettingsProvider({ children }: { children: ReactNode }) {
  const { user, loading: authLoading } = useAuth();
  const [settings, setSettings] = useState<SettingsResponse | null>(null);
  const [hfStatus, setHfStatus] = useState<HfHubStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const cacheRef = useRef<{ userId: string; settings: SettingsResponse; hfStatus: HfHubStatus } | null>(
    null,
  );

  const refresh = useCallback(async () => {
    if (!user) {
      setSettings(null);
      setHfStatus(null);
      return;
    }
    setLoading(true);
    try {
      invalidateApiCache("/settings");
      const [nextSettings, nextHfStatus] = await Promise.all([
        refreshCachedGet<SettingsResponse>("/settings", 300_000),
        refreshCachedGet<HfHubStatus>("/settings/hf-status", 15_000),
      ]);
      cacheRef.current = { userId: user.id, settings: nextSettings, hfStatus: nextHfStatus };
      setSettings(nextSettings);
      setHfStatus(nextHfStatus);
    } catch {
      /* keep last known values on transient errors */
    } finally {
      setLoading(false);
    }
  }, [user]);

  useEffect(() => {
    if (authLoading) return;
    if (!user) {
      setSettings(null);
      setHfStatus(null);
      cacheRef.current = null;
      setLoading(false);
      return;
    }
    const cached = cacheRef.current;
    if (cached?.userId === user.id) {
      setSettings(cached.settings);
      setHfStatus(cached.hfStatus);
      return;
    }
    void refresh();
  }, [user, authLoading, refresh]);

  return (
    <PlatformSettingsContext.Provider value={{ settings, hfStatus, loading, refresh }}>
      {children}
    </PlatformSettingsContext.Provider>
  );
}

export function usePlatformSettings(): PlatformSettingsState {
  const ctx = useContext(PlatformSettingsContext);
  if (!ctx) {
    throw new Error("usePlatformSettings must be used within PlatformSettingsProvider");
  }
  return ctx;
}
