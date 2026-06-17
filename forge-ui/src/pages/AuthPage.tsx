import { useState } from "react";
import { useAuth } from "@/hooks/useAuth";

export function AuthPage() {
  const { needsOnboarding, login, register } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const mode = needsOnboarding ? "register" : "login";

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    try {
      if (mode === "register") await register(email, password, name || undefined);
      else await login(email, password);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Auth failed");
    }
  };

  return (
    <div className="auth-page">
      <div className="card auth-card">
        <div className="auth-brand">
          <span className="brand-mark">◈</span>
          <h1 className="page-title" style={{ margin: 0 }}>Seiso Forge</h1>
        </div>
        <p className="page-sub">
          {needsOnboarding
            ? "Create your local admin account. This instance allows one user — your data stays on this machine."
            : "Sign in to your local workspace."}
        </p>
        <form onSubmit={submit}>
          {mode === "register" && (
            <>
              <label>Display name</label>
              <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Optional" autoComplete="name" />
            </>
          )}
          <label>Email</label>
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
          />
          <label>Password (min 8 characters)</label>
          <input
            type="password"
            required
            minLength={8}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={mode === "register" ? "new-password" : "current-password"}
          />
          {error && <p className="auth-error">{error}</p>}
          <button type="submit" className="btn btn-primary auth-submit">
            {mode === "register" ? "Create account" : "Sign in"}
          </button>
        </form>
        <p className="auth-footnote">
          Sessions are secured with HttpOnly cookies and CSRF tokens — nothing sensitive is stored in localStorage.
        </p>
      </div>
    </div>
  );
}
