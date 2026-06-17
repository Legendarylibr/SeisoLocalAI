import { useEffect, useState } from "react";
import { api, HfHubStatus } from "@/lib/api";
import { useAuth } from "@/hooks/useAuth";
import { SecurityShield } from "@/components/SecurityShield";
import { PageHeader } from "@/components/PageHeader";
import { Tabs } from "@/components/Tabs";
import { IconGlobe, IconServer, IconShield, IconUser } from "@/components/Icons";

type SettingsTab = "account" | "huggingface" | "server" | "hardening";

export function SettingsPage() {
  const { logout } = useAuth();
  const [tab, setTab] = useState<SettingsTab>("account");
  const [settings, setSettings] = useState<Awaited<ReturnType<typeof api.settings>> | null>(null);
  const [hfToken, setHfToken] = useState("");
  const [hfMsg, setHfMsg] = useState("");
  const [hfStatus, setHfStatus] = useState<HfHubStatus | null>(null);

  const refresh = () => {
    api.settings().then(setSettings).catch(console.error);
    api.hfStatus().then(setHfStatus).catch(console.error);
  };

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
    <div className="settings-page">
      <PageHeader
        title="Settings"
        subtitle="Account, integrations, server info, and your security posture at a glance."
        group="Platform"
      />

      <Tabs
        className="settings-tab-bar tab-bar-compact"
        aria-label="Settings sections"
        value={tab}
        onChange={setTab}
        items={[
          {
            id: "account",
            label: "Account",
            description: "Sign in & security score",
            icon: <IconUser size={15} />,
          },
          {
            id: "huggingface",
            label: "Hugging Face",
            description: "Hub auth & downloads",
            icon: <IconGlobe size={15} />,
            badge: settings?.hf_configured ? "Ready" : undefined,
          },
          {
            id: "server",
            label: "Server",
            description: "Bind, data dir, backend",
            icon: <IconServer size={15} />,
          },
          {
            id: "hardening",
            label: "Hardening",
            description: "Security env vars",
            icon: <IconShield size={15} />,
          },
        ]}
      />

      {tab === "account" && (
        <div className="settings-panel">
          <div className="settings-grid">
            <div className="card">
              <div className="card-head">
                <span className="card-head-icon" aria-hidden>
                  <IconUser size={18} />
                </span>
                <div className="card-head-text">
                  <h3>Local account</h3>
                  <p>Password-protected session on this machine only.</p>
                </div>
              </div>
              <button className="btn" onClick={() => logout()}>
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
        </div>
      )}

      {tab === "huggingface" && settings && (
        <div className="settings-panel">
          <div className="card">
            <div className="card-head">
              <span className="card-head-icon" aria-hidden>
                <IconGlobe size={18} />
              </span>
              <div className="card-head-text">
                <h3>Hugging Face Hub</h3>
                <p>
                  Required for gated models and publishing exports. Public downloads work without a token.
                  You can also set <code>SEISO_HF_TOKEN</code> or run{" "}
                  <code>huggingface-cli login</code> / <code>hf auth login</code>.
                </p>
              </div>
            </div>

            {hfStatus && (
              <table className="status-table">
                <tbody>
                  <tr>
                    <td>Hub reachable</td>
                    <td>
                      {hfStatus.connectivity.reachable
                        ? `Yes${hfStatus.connectivity.latency_ms != null ? ` (${hfStatus.connectivity.latency_ms} ms)` : ""}`
                        : "No"}
                    </td>
                  </tr>
                  {hfStatus.connectivity.warning && (
                    <tr>
                      <td>Token warning</td>
                      <td className="muted-text">{hfStatus.connectivity.warning}</td>
                    </tr>
                  )}
                  {hfStatus.connectivity.error && (
                    <tr>
                      <td>Hub error</td>
                      <td className="muted-text">{hfStatus.connectivity.error}</td>
                    </tr>
                  )}
                  <tr>
                    <td>Transfer backend</td>
                    <td>
                      {hfStatus.transfer.xet_available
                        ? `hf_xet (Rust)${hfStatus.transfer.high_performance ? ", high performance" : ""}`
                        : "HTTP (install hf-xet for faster downloads)"}
                    </td>
                  </tr>
                  <tr>
                    <td>Download threads</td>
                    <td className="mono">{hfStatus.transfer.num_threads}</td>
                  </tr>
                  <tr>
                    <td>Download timeout</td>
                    <td className="mono">{hfStatus.transfer.download_timeout_s}s</td>
                  </tr>
                  <tr>
                    <td>Cache dir</td>
                    <td className="mono">{hfStatus.cache_dir}</td>
                  </tr>
                  <tr>
                    <td>Ready to download</td>
                    <td>{hfStatus.ready_for_download ? "Yes" : "No"}</td>
                  </tr>
                  <tr>
                    <td>Local chat runtime</td>
                    <td>
                      {hfStatus.ready_for_local_chat
                        ? "Ready"
                        : hfStatus.ready_for_gguf_chat
                          ? "Ready (GGUF)"
                          : "Missing inference engine"}
                    </td>
                  </tr>
                  <tr>
                    <td>GGUF chat runtime</td>
                    <td>{hfStatus.ready_for_gguf_chat ? "Ready" : "Missing llama.cpp"}</td>
                  </tr>
                </tbody>
              </table>
            )}

            {hfStatus && hfStatus.transfer.hints.length > 0 && (
              <div className="env-hints" style={{ marginBottom: "0.75rem" }}>
                {hfStatus.transfer.hints.map((hint) => (
                  <div className="env-hint" key={hint}>
                    <code>{hint}</code>
                  </div>
                ))}
              </div>
            )}

            {hfStatus && hfStatus.runtime.install_hints.length > 0 && (
              <div className="env-hints" style={{ marginBottom: "0.75rem" }}>
                {hfStatus.runtime.install_hints.map((hint) => (
                  <div className="env-hint" key={hint}>
                    <code>{hint}</code>
                  </div>
                ))}
              </div>
            )}

            <table className="status-table">
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
            <div className="form-actions">
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
        </div>
      )}

      {tab === "server" && settings && (
        <div className="settings-panel">
          <div className="card">
            <div className="card-head">
              <span className="card-head-icon" aria-hidden>
                <IconServer size={18} />
              </span>
              <div className="card-head-text">
                <h3>Server configuration</h3>
                <p>Where Seiso listens and stores local data.</p>
              </div>
            </div>
            <table className="status-table">
              <tbody>
                <tr><td>Bind</td><td>{settings.host}:{settings.port}</td></tr>
                <tr><td>Data dir</td><td className="mono">{settings.data_dir}</td></tr>
                <tr><td>Backend</td><td><span className="badge">{settings.backend}</span></td></tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {tab === "hardening" && (
        <div className="settings-panel">
          <div className="card">
            <div className="card-head">
              <span className="card-head-icon" aria-hidden>
                <IconShield size={18} />
              </span>
              <div className="card-head-text">
                <h3>Hardening guide</h3>
                <p>Seiso defaults to a secure local-first posture. Enable powerful features only when you need them.</p>
              </div>
            </div>
            <div className="env-hints">
              <div className="env-hint">
                <code>SEISO_ALLOW_REMOTE=false</code>
                <span>Keep bound to localhost</span>
              </div>
              <div className="env-hint">
                <code>SEISO_ALLOW_TOOLS=false</code>
                <span>Disable web search and artifacts</span>
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
      )}
    </div>
  );
}
