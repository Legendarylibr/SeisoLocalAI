import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useAuth } from "@/hooks/useAuth";

export function SettingsPage() {
  const { user, logout } = useAuth();
  const [settings, setSettings] = useState<{ data_dir: string; backend: string; host: string; port: number } | null>(null);

  useEffect(() => {
    api.settings().then(setSettings).catch(console.error);
  }, []);

  return (
    <div>
      <h1 className="page-title">Settings</h1>
      <p className="page-sub">Profile, server monitor, and platform info.</p>

      <div className="card">
        <h3 style={{ marginBottom: "0.75rem" }}>Account</h3>
        <p>Email: {user?.email}</p>
        <p style={{ color: "var(--muted)" }}>Name: {user?.display_name || "—"}</p>
        <button className="btn" style={{ marginTop: "1rem" }} onClick={() => logout()}>
          Sign out
        </button>
      </div>

      {settings && (
        <div className="card">
          <h3 style={{ marginBottom: "0.75rem" }}>Server</h3>
          <table>
            <tbody>
              <tr><td>Bind</td><td>{settings.host}:{settings.port}</td></tr>
              <tr><td>Data dir</td><td style={{ fontFamily: "var(--mono)", fontSize: "0.85rem" }}>{settings.data_dir}</td></tr>
              <tr><td>Backend</td><td><span className="badge">{settings.backend}</span></td></tr>
            </tbody>
          </table>
        </div>
      )}

      <div className="card">
        <h3 style={{ marginBottom: "0.5rem" }}>Security defaults</h3>
        <ul style={{ color: "var(--muted)", fontSize: "0.9rem", paddingLeft: "1.25rem" }}>
          <li>Localhost binding unless SEISO_ALLOW_REMOTE=true</li>
          <li>HttpOnly session cookies + Bearer JWT</li>
          <li>Path sandboxing for all file operations</li>
          <li>Rate limiting on API routes</li>
          <li>Single-user registration lock after onboarding</li>
        </ul>
      </div>
    </div>
  );
}
