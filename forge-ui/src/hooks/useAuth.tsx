import { createContext, useCallback, useContext, useEffect, useMemo, useState, ReactNode } from "react";
import { api, AuthUser, clearLegacyToken } from "@/lib/api";
import { invalidateApiCache } from "@/lib/api/getCache";
import {
  type KeyBackup,
  persistKeyBackup,
  readStoredKeyBackup,
} from "@/lib/keyBackup";

export type { KeyBackup };

type AuthState = {
  user: AuthUser | null;
  loading: boolean;
  needsOnboarding: boolean;
  storageModeConfigured: boolean;
  storageMode: "persistent" | "ephemeral";
  /** Fresh keygen — block the app until the user confirms they wrote the nsec down. */
  keyBackup: KeyBackup | null;
  login: (nsec: string) => Promise<void>;
  register: (
    body: { generate: true } | { nsec: string },
    storageMode?: "persistent" | "ephemeral",
  ) => Promise<{ nsec?: string | null; npub?: string | null }>;
  confirmKeyBackup: () => void | Promise<void>;
  resetSession: () => Promise<void>;
  logout: () => Promise<void>;
};

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [needsOnboarding, setNeedsOnboarding] = useState(true);
  const [storageModeConfigured, setStorageModeConfigured] = useState(false);
  const [storageMode, setStorageMode] = useState<"persistent" | "ephemeral">("persistent");
  const [keyBackup, setKeyBackupState] = useState<KeyBackup | null>(() => readStoredKeyBackup());
  const [pendingUser, setPendingUser] = useState<AuthUser | null>(null);

  const setKeyBackup = useCallback((backup: KeyBackup | null) => {
    persistKeyBackup(backup);
    setKeyBackupState(backup);
  }, []);

  useEffect(() => {
    clearLegacyToken();
    (async () => {
      try {
        const status = await api.authStatus();
        setNeedsOnboarding(status.needs_onboarding);
        setStorageModeConfigured(status.storage_mode_configured);
        setStorageMode(status.storage_mode);
        const pendingBackup = readStoredKeyBackup();
        // Stay on the nsec write-down screen across refresh until Continue.
        if (pendingBackup?.nsec && pendingBackup?.npub) {
          setKeyBackupState(pendingBackup);
          setUser(null);
          return;
        }
        if (!status.needs_onboarding) {
          try {
            const me = await api.me();
            setUser(me);
          } catch {
            setUser(null);
          }
        }
      } catch {
        setUser(null);
        setNeedsOnboarding(true);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const login = useCallback(async (nsec: string) => {
    const res = await api.login(nsec);
    setUser(res.user);
    setNeedsOnboarding(false);
  }, []);

  const register = useCallback(
    async (
      body: { generate: true } | { nsec: string },
      nextStorageMode?: "persistent" | "ephemeral",
    ) => {
      const payload =
        "generate" in body && body.generate
          ? ({ generate: true } as const)
          : ({ nsec: (body as { nsec: string }).nsec } as const);
      const res = await api.register(payload, nextStorageMode);
      setNeedsOnboarding(false);
      setStorageModeConfigured(true);
      if (nextStorageMode) setStorageMode(nextStorageMode);
      const npub = res.user.npub || null;
      // Generated keys: hold the session until the user writes down their nsec.
      if (res.nsec && npub) {
        setPendingUser(res.user);
        setKeyBackup({ nsec: res.nsec, npub });
        return { nsec: res.nsec, npub };
      }
      setUser(res.user);
      return { nsec: res.nsec, npub };
    },
    [setKeyBackup],
  );

  const confirmKeyBackup = useCallback(async () => {
    if (pendingUser) {
      setUser(pendingUser);
    } else {
      try {
        const me = await api.me();
        setUser(me);
      } catch {
        setUser(null);
      }
    }
    setPendingUser(null);
    setKeyBackup(null);
  }, [pendingUser, setKeyBackup]);

  const resetSession = useCallback(async () => {
    const res = await api.resetSession("RESET");
    invalidateApiCache();
    setUser(null);
    setPendingUser(null);
    setKeyBackup(null);
    setNeedsOnboarding(res.needs_onboarding);
    const status = await api.authStatus();
    setStorageModeConfigured(status.storage_mode_configured);
    setStorageMode(status.storage_mode);
  }, [setKeyBackup]);

  const logout = useCallback(async () => {
    await api.logout();
    invalidateApiCache();
    setUser(null);
    setPendingUser(null);
    setKeyBackup(null);
  }, [setKeyBackup]);

  const value = useMemo(
    () => ({
      user,
      loading,
      needsOnboarding,
      storageModeConfigured,
      storageMode,
      keyBackup,
      login,
      register,
      confirmKeyBackup,
      resetSession,
      logout,
    }),
    [
      user,
      loading,
      needsOnboarding,
      storageModeConfigured,
      storageMode,
      keyBackup,
      login,
      register,
      confirmKeyBackup,
      resetSession,
      logout,
    ],
  );

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth outside provider");
  return ctx;
}
