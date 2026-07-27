import { useState } from "react";
import { useAuth } from "@/hooks/useAuth";
import { SeisoLogoMark } from "@/components/SeisoLogo";
import { IconLock } from "@/components/Icons";

const STORAGE_OPTIONS = [
  {
    id: "persistent" as const,
    title: "Keep my workspace",
    subtitle: "Recommended for daily use",
    detail: "Chats, models, jobs, and settings survive restarts. Stored encrypted in your local data folder.",
    badge: "Default",
  },
  {
    id: "ephemeral" as const,
    title: "Temporary session",
    subtitle: "Privacy-first, single visit",
    detail: "Everything is wiped when you sign out or close the app. Nothing is written to disk.",
    badge: null,
  },
];

export function AuthPage() {
  const {
    needsOnboarding,
    storageModeConfigured,
    keyBackup,
    login,
    register,
    confirmKeyBackup,
    resetSession,
  } = useAuth();
  const [nsec, setNsec] = useState("");
  const [storageMode, setStorageMode] = useState<"persistent" | "ephemeral">("persistent");
  const [error, setError] = useState("");
  const [resetting, setResetting] = useState(false);
  const [busy, setBusy] = useState(false);
  const mode = needsOnboarding ? "register" : "login";

  const finishRegister = async (body: { generate: true } | { nsec: string }) => {
    setBusy(true);
    setError("");
    try {
      await register(body, storageModeConfigured ? undefined : storageMode);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Auth failed");
    } finally {
      setBusy(false);
    }
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      if (mode === "register") {
        await finishRegister({ nsec: nsec.trim() });
      } else {
        await login(nsec.trim());
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Auth failed");
    } finally {
      setBusy(false);
    }
  };

  const generateAndContinue = async () => {
    await finishRegister({ generate: true });
  };

  const resetForgottenKey = async () => {
    setError("");
    const confirmed = window.confirm(
      "Start a new local Seiso session? This clears the current local account, chats, jobs, providers, and model registry entries. Downloaded model files remain on disk.",
    );
    if (!confirmed) return;
    setResetting(true);
    try {
      await resetSession();
      setNsec("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Reset failed");
    } finally {
      setResetting(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-atmosphere" aria-hidden>
        <div className="auth-orb auth-orb-a" />
        <div className="auth-orb auth-orb-b" />
        <div className="auth-grid" />
      </div>

      <div className="auth-layout">
        <div className="auth-aside">
          <div className="auth-aside-brand">
            <span className="brand-mark brand-mark-lg">
              <SeisoLogoMark className="brand-logo-img" />
            </span>
            <h1 className="auth-aside-title">Seiso Local AI</h1>
          </div>
          <p className="auth-aside-copy">
            A local-first AI workspace. Sign in with a Nostr key — your npub is your identity; models and chat stay on this machine.
          </p>
          <ul className="auth-feature-list">
            <li>
              <IconLock size={15} />
              <span>Nostr npub identity (nsec shown once on keygen, then encrypted at rest)</span>
            </li>
            <li>
              <IconLock size={15} />
              <span>HttpOnly sessions with CSRF protection</span>
            </li>
            <li>
              <IconLock size={15} />
              <span>No telemetry — nothing leaves this device unless you attest</span>
            </li>
          </ul>
        </div>

        <div className="card auth-card matte-glow">
          {keyBackup ? (
            <>
              <div className="auth-card-header">
                <h2 className="auth-card-title">Write down your nsec</h2>
                <p className="auth-card-sub">
                  This private key unlocks this Seiso instance. You will need it to sign in again.
                  Keep it secret and offline — anyone with it can control this workspace.
                </p>
              </div>
              <div className="auth-key-backup" role="status">
                <pre id="auth-nsec-reveal" className="auth-key-backup-value mono" aria-label="Your nsec">
                  {keyBackup.nsec}
                </pre>
                <p className="auth-key-backup-prompt">
                  Write this nsec down now. You will not see it again on this screen.
                </p>
                <div className="auth-key-backup-public">
                  <span className="muted-text">Public identity (npub)</span>
                  <pre id="auth-npub-reveal" className="auth-key-backup-npub mono" aria-label="Your npub">
                    {keyBackup.npub}
                  </pre>
                </div>
                <button
                  type="button"
                  className="btn btn-primary auth-submit"
                  onClick={() => void confirmKeyBackup()}
                >
                  Continue
                </button>
              </div>
            </>
          ) : (
            <>
              <div className="auth-card-header">
                <h2 className="auth-card-title">
                  {mode === "register" ? "Create your Nostr identity" : "Welcome back"}
                </h2>
                <p className="auth-card-sub">
                  {needsOnboarding
                    ? "Default: generate a fresh Nostr key. Or import an existing nsec. Your public npub identifies this instance."
                    : "Paste the nsec for this instance to unlock the workspace."}
                </p>
              </div>

              <form onSubmit={submit} className="auth-form">
                <label htmlFor="auth-nsec">
                  {mode === "register" ? "nsec (import existing)" : "nsec"}
                </label>
                <input
                  id="auth-nsec"
                  type="password"
                  required={mode === "login"}
                  minLength={mode === "login" ? 8 : 0}
                  value={nsec}
                  onChange={(e) => setNsec(e.target.value)}
                  autoComplete="off"
                  autoFocus={mode === "login"}
                  placeholder="nsec1…"
                  spellCheck={false}
                />
                {mode === "register" && !storageModeConfigured && (
                  <div className="auth-storage-section">
                    <div className="auth-storage-header">
                      <span className="auth-storage-label">How should we store your data?</span>
                      <span className="auth-storage-hint muted-text">Choose once — applies to this machine</span>
                    </div>
                    <div className="auth-storage-cards" role="radiogroup" aria-label="Storage mode">
                      {STORAGE_OPTIONS.map((opt) => {
                        const selected = storageMode === opt.id;
                        return (
                          <button
                            key={opt.id}
                            type="button"
                            role="radio"
                            aria-checked={selected}
                            className={`auth-storage-card${selected ? " auth-storage-card-selected" : ""}`}
                            onClick={() => setStorageMode(opt.id)}
                          >
                            <div className="auth-storage-card-top">
                              <span className="auth-storage-card-title">{opt.title}</span>
                              {opt.badge && <span className="auth-storage-card-badge">{opt.badge}</span>}
                            </div>
                            <span className="auth-storage-card-sub">{opt.subtitle}</span>
                            <p className="auth-storage-card-detail">{opt.detail}</p>
                            <span className="auth-storage-card-check" aria-hidden>{selected ? "✓" : ""}</span>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
                {error && <p className="auth-error" role="alert">{error}</p>}
                {mode === "register" ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                    <button
                      type="button"
                      className="btn btn-primary auth-submit"
                      disabled={busy}
                      onClick={() => void generateAndContinue()}
                    >
                      {busy ? "Working…" : "Generate key and continue"}
                    </button>
                    <button
                      type="submit"
                      className="btn auth-submit"
                      disabled={busy || !nsec.trim()}
                    >
                      Import nsec and continue
                    </button>
                  </div>
                ) : (
                  <button type="submit" className="btn btn-primary auth-submit" disabled={busy || !nsec.trim()}>
                    {busy ? "Signing in…" : "Sign in with nsec"}
                  </button>
                )}
                {mode === "login" && (
                  <button
                    type="button"
                    className="auth-reset-link"
                    onClick={resetForgottenKey}
                    disabled={resetting}
                  >
                    {resetting ? "Starting a new session..." : "Lost your nsec? Start a new session"}
                  </button>
                )}
              </form>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
