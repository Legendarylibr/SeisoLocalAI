import { useState } from "react";
import { useAuth } from "@/hooks/useAuth";
import { SeisoLogoMark } from "@/components/SeisoLogo";
import { IconLock } from "@/components/Icons";
import authBgUrl from "@/assets/auth-bg.png";

export function AuthPage() {
  const { needsOnboarding, storageModeConfigured, login, register } = useAuth();
  const [password, setPassword] = useState("");
  const [storageMode, setStorageMode] = useState<"persistent" | "ephemeral">("persistent");
  const [error, setError] = useState("");
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

  return (
    <div className="auth-page">
      <div className="auth-atmosphere" aria-hidden>
        <img src={authBgUrl} alt="" className="auth-bg-figure" draggable={false} />
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
              <fieldset className="auth-storage-choice">
                <legend>Storage mode</legend>
                <label>
                  <input
                    type="radio"
                    name="storage-mode"
                    value="persistent"
                    checked={storageMode === "persistent"}
                    onChange={() => setStorageMode("persistent")}
                  />
                  Keep my workspace across restarts
                </label>
                <label>
                  <input
                    type="radio"
                    name="storage-mode"
                    value="ephemeral"
                    checked={storageMode === "ephemeral"}
                    onChange={() => setStorageMode("ephemeral")}
                  />
                  Temporary session only
                </label>
                <p className="muted-text">
                  Persistent mode stores users, chats, models, jobs, and provider settings in your local data directory.
                </p>
              </fieldset>
            )}
            {error && <p className="auth-error" role="alert">{error}</p>}
            <button type="submit" className="btn btn-primary auth-submit">
              {mode === "register" ? "Set password and continue" : "Sign in"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
