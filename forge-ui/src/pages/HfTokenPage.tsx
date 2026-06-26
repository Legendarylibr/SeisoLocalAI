import { useState } from "react";
import { api } from "@/lib/api";
import { usePlatformSettings } from "@/context/PlatformSettingsContext";
import { SeisoLogoMark } from "@/components/SeisoLogo";
import { IconGlobe, IconLock } from "@/components/Icons";

type HfTokenPageProps = {
  onDone: () => void;
};

export function HfTokenPage({ onDone }: HfTokenPageProps) {
  const { hfStatus, refresh } = usePlatformSettings();
  const [token, setToken] = useState("");
  const [message, setMessage] = useState("");
  const [saving, setSaving] = useState(false);

  const tokenInvalid = hfStatus?.connectivity.token_invalid ?? false;

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
    <div className="auth-page hf-token-page">
      <div className="auth-atmosphere" aria-hidden>
        <div className="auth-orb auth-orb-a" />
        <div className="auth-orb auth-orb-b" />
        <div className="auth-grid" />
      </div>

      <div className="auth-layout hf-token-layout">
        <div className="auth-aside">
          <div className="auth-aside-brand">
            <span className="brand-mark brand-mark-lg">
              <IconGlobe size={22} />
            </span>
            <h1 className="auth-aside-title">Hugging Face Hub</h1>
          </div>
          <p className="auth-aside-copy">
            Connect your Hub account to unlock gated model downloads and publishing. Everything else in Seiso works without a token.
          </p>
          <ul className="auth-feature-list">
            <li>
              <IconLock size={15} />
              <span>Needed for gated repos and publishing exports</span>
            </li>
            <li>
              <IconGlobe size={15} />
              <span>Public GGUF downloads work without a token</span>
            </li>
            <li>
              <IconLock size={15} />
              <span>Stored encrypted on this machine only</span>
            </li>
          </ul>
        </div>

        <div className="card auth-card matte-glow hf-token-card">
          <div className="auth-card-header">
            <span className="brand-mark" style={{ margin: "0 auto 0.75rem" }}>
              <SeisoLogoMark className="brand-logo-img" />
            </span>
            <h2 className="auth-card-title">Add your HF token</h2>
            <p className="auth-card-sub">
              Optional — skipping does not block public model downloads. Upload to the Hub always requires a valid token.
            </p>
          </div>

          {tokenInvalid && (
            <p className="auth-error" role="alert">
              Your saved token was rejected. Enter a new one or run <code>hf auth login</code>.
            </p>
          )}

          {hfStatus && (
            <table className="status-table" style={{ marginBottom: "0.85rem" }}>
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
                  <td>Public downloads</td>
                  <td>{hfStatus.ready_for_download ? "Ready" : "Unavailable"}</td>
                </tr>
                <tr>
                  <td>Gated downloads &amp; upload</td>
                  <td>
                    {hfStatus.connectivity.token_valid
                      ? `Ready${hfStatus.connectivity.token_username ? ` (${hfStatus.connectivity.token_username})` : ""}`
                      : "Needs a valid token"}
                  </td>
                </tr>
              </tbody>
            </table>
          )}

          <form
            className="auth-form"
            onSubmit={(e) => {
              e.preventDefault();
              void save();
            }}
          >
            <label htmlFor="hf-token-page-input">API token</label>
            <input
              id="hf-token-page-input"
              type="password"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="hf_..."
              autoComplete="off"
              autoFocus
            />
            <p className="muted-text" style={{ fontSize: "0.78rem", marginTop: "0.35rem" }}>
              Create a token at{" "}
              <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
                huggingface.co/settings/tokens
              </a>{" "}
              with read access for downloads and write access for publishing.
            </p>

            {message && (
              <p
                className={`hf-token-message${message.includes("saved") ? " hf-token-message-ok" : " hf-token-message-err"}`}
                role="alert"
              >
                {message}
              </p>
            )}

            <div className="form-actions hf-token-actions">
              <button className="btn btn-primary auth-submit" type="submit" disabled={!token.trim() || saving}>
                {saving ? "Saving..." : "Save and continue"}
              </button>
              <button className="btn" type="button" onClick={onDone} disabled={saving}>
                Skip for now
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}