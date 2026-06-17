import { useEffect, useState } from "react";
import { api, HfHubStatus } from "@/lib/api";
import { IconGlobe } from "@/components/Icons";

type HfTokenPromptProps = {
  onDone: () => void;
};

export function HfTokenPrompt({ onDone }: HfTokenPromptProps) {
  const [token, setToken] = useState("");
  const [status, setStatus] = useState<HfHubStatus | null>(null);
  const [message, setMessage] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    api.hfStatus().then(setStatus).catch(() => setStatus(null));
  }, []);

  const save = async () => {
    if (!token.trim()) return;
    setSaving(true);
    setMessage("");
    try {
      await api.saveHfToken(token.trim());
      setMessage("Hugging Face token saved.");
      onDone();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Could not save token");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-atmosphere" aria-hidden>
        <div className="auth-orb auth-orb-a" />
        <div className="auth-orb auth-orb-b" />
        <div className="auth-grid" />
      </div>

      <div className="auth-layout hf-token-layout">
        <div className="card auth-card matte-glow hf-token-card">
          <div className="card-head">
            <span className="card-head-icon" aria-hidden>
              <IconGlobe size={18} />
            </span>
            <div className="card-head-text">
              <h3>Hugging Face Hub</h3>
              <p>
                Add a token now for gated models, higher anonymous rate limits, and publishing.
                Public downloads still work without one.
              </p>
            </div>
          </div>

          {status && (
            <table className="status-table">
              <tbody>
                <tr>
                  <td>Hub reachable</td>
                  <td>
                    {status.connectivity.reachable
                      ? `Yes${status.connectivity.latency_ms != null ? ` (${status.connectivity.latency_ms} ms)` : ""}`
                      : "No"}
                  </td>
                </tr>
                <tr>
                  <td>Ready to download</td>
                  <td>
                    {status.ready_for_download
                      ? status.connectivity.token_valid
                        ? "Yes (authenticated)"
                        : "Yes (public models, no token)"
                      : "No"}
                  </td>
                </tr>
                <tr>
                  <td>Cache dir</td>
                  <td className="mono">{status.cache_dir}</td>
                </tr>
              </tbody>
            </table>
          )}

          <label htmlFor="hf-token-prompt">API token</label>
          <input
            id="hf-token-prompt"
            type="password"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            placeholder="hf_..."
            autoComplete="off"
            autoFocus
          />

          <div className="form-actions hf-token-actions">
            <button className="btn btn-primary" type="button" onClick={save} disabled={!token.trim() || saving}>
              {saving ? "Saving..." : "Save token"}
            </button>
            <button className="btn" type="button" onClick={onDone} disabled={saving}>
              Skip for now
            </button>
          </div>

          {message && <p className="muted-text hf-token-message">{message}</p>}
        </div>
      </div>
    </div>
  );
}
