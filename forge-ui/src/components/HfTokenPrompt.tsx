import { useState } from "react";
import { api } from "@/lib/api";
import { usePlatformSettings } from "@/context/PlatformSettingsContext";
import { IconGlobe } from "@/components/Icons";

type HfTokenPromptProps = {
  onDone: () => void;
};

export function HfTokenPrompt({ onDone }: HfTokenPromptProps) {
  const { hfStatus, refresh } = usePlatformSettings();
  const [token, setToken] = useState("");
  const [message, setMessage] = useState("");
  const [saving, setSaving] = useState(false);

  const save = async () => {
    if (!token.trim()) return;
    setSaving(true);
    setMessage("");
    try {
      await api.saveHfToken(token.trim());
      await refresh();
      setMessage("Hugging Face token saved.");
      onDone();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Could not save token");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="hf-token-prompt-title">
      <button type="button" className="modal-backdrop" onClick={onDone} aria-label="Dismiss" />
      <div className="modal-panel card matte-glow hf-token-card">
        <div className="card-head">
          <span className="card-head-icon" aria-hidden>
            <IconGlobe size={18} />
          </span>
          <div className="card-head-text">
            <h3 id="hf-token-prompt-title">Connect Hugging Face</h3>
            <p>
              Optional — add a token for gated models, higher rate limits, and publishing.
              Public GGUF downloads work without one.
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
                  <td>Cache dir</td>
                  <td className="mono">{hfStatus.cache_dir}</td>
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

          {message && (
            <p className={`hf-token-message${message.includes("saved") ? " hf-token-message-ok" : " hf-token-message-err"}`}>
              {message}
            </p>
          )}
      </div>
    </div>
  );
}
