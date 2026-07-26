import { createContext, useCallback, useContext, useEffect, useMemo, useState, ReactNode } from "react";
import { api, AuthUser, clearLegacyToken } from "@/lib/api";

export type KeyBackup = {
  npub: string;
};

type AuthState = {
  user: AuthUser | null;
  loading: boolean;
  needsOnboarding: boolean;
  storageModeConfigured: boolean;
  storageMode: "persistent" | "ephemeral";
  /** Fresh keygen — block the app until the user confirms they wrote the npub down. */
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

const KEY_BACKUP_STORAGE = "seiso_key_backup";

function readStoredKeyBackup(): KeyBackup | null {
  try {
    const raw = sessionStorage.getItem(KEY_BACKUP_STORAGE);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as KeyBackup;
    return parsed?.npub ? { npub: parsed.npub } : null;
  } catch {
    return null;
  }
}

function persistKeyBackup(backup: KeyBackup | null) {
  try {
    if (backup?.npub) {
      sessionStorage.setItem(KEY_BACKUP_STORAGE, JSON.stringify(backup));
    } else {
      sessionStorage.removeItem(KEY_BACKUP_STORAGE);
    }
  } catch {
    /* ignore quota / private mode */
  }
}

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
        // Stay on the npub write-down screen across refresh until Continue.
        if (pendingBackup?.npub) {
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
      // Generated keys: hold the session until the user writes down their npub.
      if (res.nsec && npub) {
        setPendingUser(res.user);
        setKeyBackup({ npub });
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
