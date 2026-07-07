import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "@/lib/api";
import { usePlatformSettings } from "@/context/PlatformSettingsContext";
import { useAuth } from "@/hooks/useAuth";
import { SecurityShield } from "@/components/SecurityShield";
import { PageHeader } from "@/components/PageHeader";
import { Tabs } from "@/components/Tabs";
import { IconGlobe, IconServer, IconShield, IconUser } from "@/components/Icons";

type SettingsTab = "account" | "huggingface" | "server" | "hardening";

function ggufSidecarRuntimeLabel(
  ready: boolean,
  runtime: {
    ollama_ready?: boolean;
    llamaswap_ready?: boolean;
    llamaswap_engine?: string | null;
  },
): string {
  if (ready) {
    if (runtime.ollama_ready) {
      return runtime.llamaswap_engine === "ollama"
        ? "Ready (Ollama)"
        : "Ready (Ollama sidecar)";
    }
    if (runtime.llamaswap_ready) {
      return "Ready (llama-swap fallback)";
    }
    return "Ready";
  }
  if (runtime.ollama_ready) {
    return "Ollama up — sidecar routing unavailable";
  }
  return "Missing Ollama sidecar";
}

export function SettingsPage() {
  const { logout } = useAuth();
  const [searchParams] = useSearchParams();
  const [tab, setTab] = useState<SettingsTab>(() => {
    const requested = searchParams.get("tab");
    if (requested === "huggingface" || requested === "server" || requested === "hardening" || requested === "account") {
      return requested;
    }
    return "account";
  });
  const { settings, hfStatus, refresh } = usePlatformSettings();
  const [hfToken, setHfToken] = useState("");
  const [hfMsg, setHfMsg] = useState("");

  useEffect(() => {
    const requested = searchParams.get("tab");
    if (requested === "huggingface" || requested === "server" || requested === "hardening" || requested === "account") {
      setTab(requested);
    }
  }, [searchParams]);

  const saveToken = async () => {
    if (!hfToken.trim()) return;
    try {
      await api.saveHfToken(hfToken.trim());
      setHfToken("");
      setHfMsg("Token saved (encrypted locally).");
      await refresh();
    } catch (err) {
      setHfMsg((err as Error).message);
    }
  };

  const clearToken = async () => {
    try {
      await api.clearHfToken();
      setHfMsg("Saved token cleared.");
      await refresh();
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
            description: "Bind, data dir, backends",
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
                  {hfStatus.auth.token_invalid && (
                    <tr>
                      <td>Token status</td>
                      <td className="muted-text">Saved token is invalid — update it below or run <code>hf auth login</code>.</td>
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
                    <td>
                      {hfStatus.ready_for_download
                        ? hfStatus.connectivity.token_valid
                          ? "Yes (authenticated)"
                          : "Yes (public models, no token)"
                        : "No"}
                    </td>
                  </tr>
                  <tr>
                    <td>Ready to upload</td>
                    <td>{hfStatus.ready_for_upload ? "Yes (valid token)" : "No — token required for publishing"}</td>
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
                    <td>
                      {hfStatus
                        ? ggufSidecarRuntimeLabel(
                            hfStatus.ready_for_gguf_chat,
                            hfStatus.runtime,
                          )
                        : "—"}
                    </td>
                  </tr>
                  {hfStatus?.runtime.llamaswap_engine && (
                    <tr>
                      <td>GGUF sidecar engine</td>
                      <td>{hfStatus.runtime.llamaswap_engine}</td>
                    </tr>
                  )}
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
                  <td>Token configured</td>
                  <td>{settings.hf_configured ? "Yes" : "No — public downloads still work"}</td>
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
                <tr><td>Training backend</td><td><span className="badge">{settings.training_backend}</span></td></tr>
                <tr>
                  <td>Inference engines</td>
                  <td>
                    {settings.inference_backends.length
                      ? settings.inference_backends.map((b) => (
                          <span key={b} className="badge" style={{ marginRight: "0.35rem" }}>{b}</span>
                        ))
                      : "—"}
                  </td>
                </tr>
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
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
