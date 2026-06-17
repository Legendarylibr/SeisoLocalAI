import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useAuth } from "@/hooks/useAuth";
import { SecurityShield } from "@/components/SecurityShield";
import { PageHeader } from "@/components/PageHeader";

export function SettingsPage() {
  const { logout } = useAuth();
  const [settings, setSettings] = useState<Awaited<ReturnType<typeof api.settings>> | null>(null);
  const [hfToken, setHfToken] = useState("");
  const [hfMsg, setHfMsg] = useState("");

  const refresh = () => api.settings().then(setSettings).catch(console.error);

  useEffect(() => {
    refresh();
  }, []);

  const saveToken = async () => {
    if (!hfToken.trim()) return;
    try {
      await api.saveHfToken(hfToken.trim());
      setHfToken("");
      setHfMsg("Token saved (encrypted locally).");
      refresh();
    } catch (err) {
      setHfMsg((err as Error).message);
    }
  };

  const clearToken = async () => {
    try {
      await api.clearHfToken();
      setHfMsg("Saved token cleared.");
      refresh();
    } catch (err) {
      setHfMsg((err as Error).message);
    }
  };

  return (
    <div>
      <PageHeader
        title="Settings"
        subtitle="Account, server info, and your security posture at a glance."
      />

      <div className="settings-grid">
        <div className="card">
          <h3 className="section-title">Account</h3>
          <p className="muted-text">Password-protected local account</p>
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
          <h3 className="section-title">Hugging Face</h3>
          <p className="muted-text" style={{ marginBottom: "0.75rem" }}>
            Required to publish Seiso export outputs. You can also set <code>SEISO_HF_TOKEN</code> or run{" "}
            <code>huggingface-cli login</code> / <code>hf auth login</code>.
          </p>
          <table style={{ marginBottom: "0.75rem" }}>
            <tbody>
              <tr>
                <td>Auth ready</td>
                <td>{settings.hf_configured ? "Yes" : "No"}</td>
              </tr>
              <tr>
                <td>CLI available</td>
                <td>{settings.hf_auth.cli_available ? settings.hf_auth.cli_binary : "Not found"}</td>
              </tr>
              <tr>
                <td>CLI logged in</td>
                <td>{settings.hf_auth.cli_logged_in ? "Yes" : "No"}</td>
              </tr>
              <tr>
                <td>Saved token</td>
                <td>{settings.hf_auth.user_token_saved ? "Yes" : "No"}</td>
              </tr>
              <tr>
                <td>Sources</td>
                <td>{settings.hf_auth.token_sources.join(", ") || "—"}</td>
              </tr>
            </tbody>
          </table>
          <label>API token</label>
          <input
            type="password"
            value={hfToken}
            onChange={(e) => setHfToken(e.target.value)}
            placeholder="hf_…"
            autoComplete="off"
          />
          <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.75rem" }}>
            <button className="btn btn-primary" onClick={saveToken} disabled={!hfToken.trim()}>
              Save token
            </button>
            {settings.hf_auth.user_token_saved && (
              <button className="btn" onClick={clearToken}>
                Clear saved token
              </button>
            )}
          </div>
          {hfMsg && <p className="muted-text" style={{ marginTop: "0.5rem" }}>{hfMsg}</p>}
        </div>
      )}

      {settings && (
        <div className="card">
          <h3 className="section-title">Server</h3>
          <table>
            <tbody>
              <tr><td>Bind</td><td>{settings.host}:{settings.port}</td></tr>
              <tr><td>Data dir</td><td className="mono">{settings.data_dir}</td></tr>
              <tr><td>Backend</td><td><span className="badge">{settings.backend}</span></td></tr>
            </tbody>
          </table>
        </div>
      )}

      <div className="card">
        <h3 className="section-title">Hardening guide</h3>
        <p className="muted-text" style={{ marginBottom: "0.75rem" }}>
          Seiso defaults to a secure local-first posture. Enable powerful features only when you need them.
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
