import { useState } from "react";
import { useAuth } from "@/hooks/useAuth";

export function AuthPage() {
  const { needsOnboarding, login, register } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [mode, setMode] = useState<"login" | "register">(needsOnboarding ? "register" : "login");

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
        <h1 className="page-title">Seiso Forge</h1>
        <p className="page-sub">
          {needsOnboarding
            ? "Create your local admin account to get started."
            : "Sign in to your local workspace."}
        </p>
        <form onSubmit={submit}>
          {mode === "register" && (
            <>
              <label>Display name</label>
              <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Optional" />
            </>
          )}
          <label>Email</label>
          <input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
          <label>Password (min 8 chars)</label>
          <input
            type="password"
            required
            minLength={8}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          {error && <p style={{ color: "var(--danger)", marginBottom: "0.75rem" }}>{error}</p>}
          <button type="submit" className="btn btn-primary" style={{ width: "100%" }}>
            {mode === "register" ? "Create account" : "Sign in"}
          </button>
        </form>
        {!needsOnboarding && (
          <p style={{ marginTop: "1rem", fontSize: "0.85rem", color: "var(--muted)" }}>
            {mode === "login" ? "First time? " : "Have an account? "}
            <button
              type="button"
              className="btn"
              style={{ padding: 0, border: "none", background: "none", color: "var(--accent)" }}
              onClick={() => setMode(mode === "login" ? "register" : "login")}
            >
              {mode === "login" ? "Register" : "Sign in"}
            </button>
          </p>
        )}
      </div>
    </div>
  );
}
