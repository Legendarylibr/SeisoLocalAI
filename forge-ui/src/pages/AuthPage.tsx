import { useState } from "react";
import { useAuth } from "@/hooks/useAuth";
import { SeisoLogoMark } from "@/components/SeisoLogo";
import { IconLock } from "@/components/Icons";

const STORAGE_OPTIONS = [
  {
    id: "persistent" as const,
    title: "Keep my workspace",
    subtitle: "Recommended for daily use",
    detail: "Chats, models, jobs, and settings survive restarts. Stored encrypted in your local data folder.",
    badge: "Default",
  },
  {
    id: "ephemeral" as const,
    title: "Temporary session",
    subtitle: "Privacy-first, single visit",
    detail: "Everything is wiped when you sign out or close the app. Nothing is written to disk.",
    badge: null,
  },
];

export function AuthPage() {
  const { needsOnboarding, storageModeConfigured, login, register, resetSession } = useAuth();
  const [password, setPassword] = useState("");
  const [storageMode, setStorageMode] = useState<"persistent" | "ephemeral">("persistent");
  const [error, setError] = useState("");
  const [resetting, setResetting] = useState(false);
  const mode = needsOnboarding ? "register" : "login";

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    try {
      if (mode === "register") await register(password, storageModeConfigured ? undefined : storageMode);
      else await login(password);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Auth failed");
    }
  };

  const resetForgottenPassword = async () => {
    setError("");
    const confirmed = window.confirm(
      "Start a new local Seiso session? This clears the current local account, chats, jobs, providers, and model registry entries. Downloaded model files remain on disk.",
    );
    if (!confirmed) return;
    setResetting(true);
    try {
      await resetSession();
      setPassword("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Reset failed");
    } finally {
      setResetting(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-atmosphere" aria-hidden>
        <div className="auth-orb auth-orb-a" />
        <div className="auth-orb auth-orb-b" />
        <div className="auth-grid" />
      </div>

      <div className="auth-layout">
        <div className="auth-aside">
          <div className="auth-aside-brand">
            <span className="brand-mark brand-mark-lg">
              <SeisoLogoMark className="brand-logo-img" />
            </span>
            <h1 className="auth-aside-title">Seiso Local AI</h1>
          </div>
          <p className="auth-aside-copy">
            A local-first AI workspace. Models, training, and chat stay on your machine — encrypted, private, and under your control.
          </p>
          <ul className="auth-feature-list">
            <li>
              <IconLock size={15} />
              <span>HttpOnly sessions with CSRF protection</span>
            </li>
            <li>
              <IconLock size={15} />
              <span>Encrypted storage for chat and API keys</span>
            </li>
            <li>
              <IconLock size={15} />
              <span>No telemetry — nothing leaves this device</span>
            </li>
          </ul>
        </div>

        <div className="card auth-card matte-glow">
          <div className="auth-card-header">
            <h2 className="auth-card-title">
              {mode === "register" ? "Create your password" : "Welcome back"}
            </h2>
            <p className="auth-card-sub">
              {needsOnboarding
                ? "One account per machine. Set a password to secure this instance."
                : "Enter your password to unlock the workspace."}
            </p>
          </div>
          <form onSubmit={submit} className="auth-form">
            <label htmlFor="auth-password">
              {mode === "register" ? "Password (min 8 characters)" : "Password"}
            </label>
            <input
              id="auth-password"
              type="password"
              required
              minLength={mode === "register" ? 8 : 1}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete={mode === "register" ? "new-password" : "current-password"}
              autoFocus
              placeholder="Enter password"
            />
            {mode === "register" && !storageModeConfigured && (
              <div className="auth-storage-section">
                <div className="auth-storage-header">
                  <span className="auth-storage-label">How should we store your data?</span>
                  <span className="auth-storage-hint muted-text">Choose once — applies to this machine</span>
                </div>
                <div className="auth-storage-cards" role="radiogroup" aria-label="Storage mode">
                  {STORAGE_OPTIONS.map((opt) => {
                    const selected = storageMode === opt.id;
                    return (
                      <button
                        key={opt.id}
                        type="button"
                        role="radio"
                        aria-checked={selected}
                        className={`auth-storage-card${selected ? " auth-storage-card-selected" : ""}`}
                        onClick={() => setStorageMode(opt.id)}
                      >
                        <div className="auth-storage-card-top">
                          <span className="auth-storage-card-title">{opt.title}</span>
                          {opt.badge && <span className="auth-storage-card-badge">{opt.badge}</span>}
                        </div>
                        <span className="auth-storage-card-sub">{opt.subtitle}</span>
                        <p className="auth-storage-card-detail">{opt.detail}</p>
                        <span className="auth-storage-card-check" aria-hidden>{selected ? "✓" : ""}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
            {error && <p className="auth-error" role="alert">{error}</p>}
            <button type="submit" className="btn btn-primary auth-submit">
              {mode === "register" ? "Set password and continue" : "Sign in"}
            </button>
            {mode === "login" && (
              <button
                type="button"
                className="auth-reset-link"
                onClick={resetForgottenPassword}
                disabled={resetting}
              >
                {resetting ? "Starting a new session..." : "Forgot password? Start a new session"}
              </button>
            )}
          </form>
        </div>
      </div>
    </div>
  );
}
