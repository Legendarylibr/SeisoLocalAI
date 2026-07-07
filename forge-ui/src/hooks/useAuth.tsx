import { createContext, useCallback, useContext, useEffect, useMemo, useState, ReactNode } from "react";
import { api, AuthUser, clearLegacyToken } from "@/lib/api";

type AuthState = {
  user: AuthUser | null;
  loading: boolean;
  needsOnboarding: boolean;
  storageModeConfigured: boolean;
  storageMode: "persistent" | "ephemeral";
  login: (password: string) => Promise<void>;
  register: (password: string, storageMode?: "persistent" | "ephemeral") => Promise<void>;
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

  useEffect(() => {
    clearLegacyToken();
    (async () => {
      try {
        const status = await api.authStatus();
        setNeedsOnboarding(status.needs_onboarding);
        setStorageModeConfigured(status.storage_mode_configured);
        setStorageMode(status.storage_mode);
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

  const login = useCallback(async (password: string) => {
    const res = await api.login(password);
    setUser(res.user);
    setNeedsOnboarding(false);
  }, []);

  const register = useCallback(async (password: string, nextStorageMode?: "persistent" | "ephemeral") => {
    const res = await api.register(password, nextStorageMode);
    setUser(res.user);
    setNeedsOnboarding(false);
    setStorageModeConfigured(true);
    if (nextStorageMode) setStorageMode(nextStorageMode);
  }, []);

  const resetSession = useCallback(async () => {
    const res = await api.resetSession("RESET");
    setUser(null);
    setNeedsOnboarding(res.needs_onboarding);
    const status = await api.authStatus();
    setStorageModeConfigured(status.storage_mode_configured);
    setStorageMode(status.storage_mode);
  }, []);

  const logout = useCallback(async () => {
    await api.logout();
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({
      user,
      loading,
      needsOnboarding,
      storageModeConfigured,
      storageMode,
      login,
      register,
      resetSession,
      logout,
    }),
    [
      user,
      loading,
      needsOnboarding,
      storageModeConfigured,
      storageMode,
      login,
      register,
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
