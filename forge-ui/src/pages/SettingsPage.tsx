import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useAuth } from "@/hooks/useAuth";
import { SecurityShield } from "@/components/SecurityShield";

export function SettingsPage() {
  const { user, logout } = useAuth();
  const [settings, setSettings] = useState<Awaited<ReturnType<typeof api.settings>> | null>(null);

  useEffect(() => {
    api.settings().then(setSettings).catch(console.error);
  }, []);

  return (
    <div>
      <h1 className="page-title">Settings</h1>
      <p className="page-sub">Account, server info, and your security posture at a glance.</p>

      <div className="settings-grid">
        <div className="card">
          <h3 className="section-title">Account</h3>
          <p>Email: {user?.email}</p>
          <p className="muted-text">Name: {user?.display_name || "—"}</p>
          <button className="btn" style={{ marginTop: "1rem" }} onClick={() => logout()}>
            Sign out
          </button>
          <p className="muted-text" style={{ marginTop: "0.75rem", fontSize: "0.8rem" }}>
            Sessions use HttpOnly cookies with CSRF protection — tokens are never stored in the browser.
          </p>
        </div>

        {settings && (
          <div className="card">
            <SecurityShield security={settings.security} />
          </div>
        )}
      </div>

      {settings && (
        <div className="card">
          <h3 className="section-title">Server</h3>
          <table>
            <tbody>
              <tr><td>Bind</td><td>{settings.host}:{settings.port}</td></tr>
              <tr><td>Data dir</td><td className="mono">{settings.data_dir}</td></tr>
              <tr><td>Backend</td><td><span className="badge">{settings.backend}</span></td></tr>
              <tr><td>Hugging Face</td><td>{settings.hf_configured ? "Configured" : "Not set"}</td></tr>
            </tbody>
          </table>
        </div>
      )}

      <div className="card">
        <h3 className="section-title">Hardening guide</h3>
        <p className="muted-text" style={{ marginBottom: "0.75rem" }}>
          Seiso Forge defaults to a secure local-first posture. Enable powerful features only when you need them.
        </p>
        <div className="env-hints">
          <div className="env-hint">
            <code>SEISO_ALLOW_REMOTE=false</code>
            <span>Keep bound to localhost</span>
          </div>
          <div className="env-hint">
            <code>SEISO_ALLOW_TOOLS=false</code>
            <span>Disable web search, artifacts, MCP</span>
          </div>
          <div className="env-hint">
            <code>SEISO_ALLOW_CODE_EXEC=false</code>
            <span>Disable sandboxed Python execution</span>
          </div>
          <div className="env-hint">
            <code>SEISO_AUTODEFENSE_ENABLED=true</code>
            <span>Scan prompts for injection attacks</span>
          </div>
        </div>
      </div>
    </div>
  );
}
