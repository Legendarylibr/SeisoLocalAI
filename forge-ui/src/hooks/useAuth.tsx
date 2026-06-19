import { createContext, useCallback, useContext, useEffect, useMemo, useState, ReactNode } from "react";
import { api, AuthUser, clearLegacyToken } from "@/lib/api";

type AuthState = {
  user: AuthUser | null;
  loading: boolean;
  needsOnboarding: boolean;
  login: (password: string) => Promise<void>;
  register: (password: string) => Promise<void>;
  logout: () => Promise<void>;
};

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [needsOnboarding, setNeedsOnboarding] = useState(false);

  useEffect(() => {
    clearLegacyToken();
    (async () => {
      try {
        const status = await api.authStatus();
        setNeedsOnboarding(status.needs_onboarding);
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

  const register = useCallback(async (password: string) => {
    const res = await api.register(password);
    setUser(res.user);
    setNeedsOnboarding(false);
  }, []);

  const logout = useCallback(async () => {
    await api.logout();
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({ user, loading, needsOnboarding, login, register, logout }),
    [user, loading, needsOnboarding, login, register, logout],
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
