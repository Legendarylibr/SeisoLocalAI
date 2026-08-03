import { useMemo, useState } from "react";
import { useAuth } from "@/hooks/useAuth";
import { downloadNip49KeyBackup } from "@/lib/keyBackup";
import { looksLikeNcryptsec, resolveSecretToNsec } from "@/lib/nip49";
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
  const [importPassphrase, setImportPassphrase] = useState("");
  const [backupPassphrase, setBackupPassphrase] = useState("");
  const [backupPassphraseConfirm, setBackupPassphraseConfirm] = useState("");
  const [storageMode, setStorageMode] = useState<"persistent" | "ephemeral">("persistent");
  const [error, setError] = useState("");
  const [resetting, setResetting] = useState(false);
  const [busy, setBusy] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const mode = needsOnboarding ? "register" : "login";
  const needsImportPassphrase = useMemo(() => looksLikeNcryptsec(nsec), [nsec]);

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

  const resolveImportSecret = async (): Promise<string> => {
    return resolveSecretToNsec(
      nsec.trim(),
      needsImportPassphrase ? importPassphrase : undefined,
    );
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      const secret = await resolveImportSecret();
      if (mode === "register") {
        await finishRegister({ nsec: secret });
      } else {
        await login(secret);
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

  const downloadEncryptedBackup = async () => {
    if (!keyBackup) return;
    setError("");
    if (backupPassphrase.length < 8) {
      setError("Passphrase must be at least 8 characters");
      return;
    }
    if (backupPassphrase !== backupPassphraseConfirm) {
      setError("Passphrases do not match");
      return;
    }
    setDownloading(true);
    try {
      await downloadNip49KeyBackup(keyBackup, { passphrase: backupPassphrase });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Download failed");
    } finally {
      setDownloading(false);
    }
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
      setImportPassphrase("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Reset failed");
    } finally {
      setResetting(false);
    }
  };

  const recoveryKeyField = (
    <>
      <label htmlFor="auth-nsec">Recovery key</label>
      <input
        id="auth-nsec"
        type="password"
        required={mode === "login"}
        minLength={mode === "login" ? 8 : 0}
        value={nsec}
        onChange={(e) => setNsec(e.target.value)}
        autoComplete="off"
        autoFocus={mode === "login"}
        placeholder="Paste your saved recovery key"
        spellCheck={false}
      />
      <p className="auth-field-hint muted-text">
        A long string that usually starts with <span className="mono">nsec1</span>
        {needsImportPassphrase ? (
          <>
            {" "}
            or an encrypted <span className="mono">ncryptsec1</span> backup.
          </>
        ) : (
          <>.</>
        )}
      </p>
      {needsImportPassphrase && (
        <>
          <label htmlFor="auth-import-passphrase">Backup passphrase</label>
          <input
            id="auth-import-passphrase"
            type="password"
            required
            value={importPassphrase}
            onChange={(e) => setImportPassphrase(e.target.value)}
            autoComplete="current-password"
            placeholder="Passphrase you chose for the encrypted backup"
          />
        </>
      )}
    </>
  );

  const storagePicker =
    mode === "register" && !storageModeConfigured ? (
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
                <span className="auth-storage-card-check" aria-hidden>
                  {selected ? "✓" : ""}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    ) : null;

  return (
    <div className="auth-page">
      <div className="auth-atmosphere" aria-hidden>
        <div className="auth-orb auth-orb-a" />
        <div className="auth-orb auth-orb-b" />
        <div className="auth-grid" />
      </div>

      <div className="auth-layout">
        <div className="auth-aside">
          <div className="auth-aside-brand auth-aside-brand-wordmark">
            <SeisoLogoMark className="auth-wordmark" />
            <h1 className="auth-aside-title">Seiso Local AI</h1>
          </div>
          <p className="auth-aside-copy">
            A local-first AI workspace. Create a private account key on this machine — models and chat
            stay here. No cloud signup.
          </p>
          <ul className="auth-feature-list">
            <li>
              <IconLock size={15} />
              <span>Private recovery key — shown once, then stored encrypted on this device</span>
            </li>
            <li>
              <IconLock size={15} />
              <span>Optional encrypted backup file you can keep offline</span>
            </li>
            <li>
              <IconLock size={15} />
              <span>Secure browser sessions with CSRF protection</span>
            </li>
            <li>
              <IconLock size={15} />
              <span>No telemetry — nothing leaves this device unless you choose to share</span>
            </li>
          </ul>
          <p className="auth-aside-footnote muted-text">
            Under the hood your account uses open Nostr key formats (
            <span className="mono">nsec</span> / <span className="mono">npub</span>). You do not need a
            Nostr app or relay to use Seiso.
          </p>
        </div>

        <div className="card auth-card matte-glow">
          {keyBackup ? (
            <>
              <div className="auth-card-header">
                <h2 className="auth-card-title">Save your recovery key</h2>
                <p className="auth-card-sub">
                  This is the only way to sign back in later. Treat it like a password-manager secret —
                  anyone with it can control this workspace.
                </p>
              </div>
              <div className="auth-key-backup" role="status">
                <span className="auth-storage-label">Recovery key (private)</span>
                <pre
                  id="auth-nsec-reveal"
                  className="auth-key-backup-value mono"
                  aria-label="Your recovery key"
                >
                  {keyBackup.nsec}
                </pre>
                <p className="auth-key-backup-prompt">
                  Write this down or store it in a password manager now. You will not see it again on
                  this screen.
                </p>
                <div className="auth-key-backup-public">
                  <span className="muted-text">Public ID (safe to share)</span>
                  <pre
                    id="auth-npub-reveal"
                    className="auth-key-backup-npub mono"
                    aria-label="Your public ID"
                  >
                    {keyBackup.npub}
                  </pre>
                </div>
                <div className="auth-key-backup-encrypt">
                  <span className="auth-storage-label">Optional: download encrypted backup</span>
                  <p className="auth-key-backup-download-hint muted-text">
                    Creates a passphrase-locked file. The file never contains the raw recovery key —
                    remember the passphrase separately.
                  </p>
                  <label htmlFor="auth-backup-passphrase">Backup passphrase</label>
                  <input
                    id="auth-backup-passphrase"
                    type="password"
                    value={backupPassphrase}
                    onChange={(e) => setBackupPassphrase(e.target.value)}
                    autoComplete="new-password"
                    minLength={8}
                    placeholder="At least 8 characters"
                  />
                  <label htmlFor="auth-backup-passphrase-confirm">Confirm passphrase</label>
                  <input
                    id="auth-backup-passphrase-confirm"
                    type="password"
                    value={backupPassphraseConfirm}
                    onChange={(e) => setBackupPassphraseConfirm(e.target.value)}
                    autoComplete="new-password"
                    minLength={8}
                    placeholder="Repeat passphrase"
                  />
                </div>
                <details className="auth-tech-details">
                  <summary>Technical names</summary>
                  <p className="muted-text">
                    Recovery key = Nostr <span className="mono">nsec</span>. Public ID ={" "}
                    <span className="mono">npub</span>. Encrypted backup = NIP-49{" "}
                    <span className="mono">ncryptsec</span>. Same bytes as before — only the labels
                    changed.
                  </p>
                </details>
                {error && (
                  <p className="auth-error" role="alert">
                    {error}
                  </p>
                )}
                <div className="auth-key-backup-actions">
                  <button
                    type="button"
                    className="btn auth-submit"
                    disabled={downloading || backupPassphrase.length < 8}
                    onClick={() => void downloadEncryptedBackup()}
                  >
                    {downloading ? "Encrypting…" : "Download encrypted .txt"}
                  </button>
                  <button
                    type="button"
                    className="btn btn-primary auth-submit"
                    onClick={() => void confirmKeyBackup()}
                  >
                    I saved my recovery key — continue
                  </button>
                </div>
              </div>
            </>
          ) : (
            <>
              <div className="auth-card-header">
                <h2 className="auth-card-title">
                  {mode === "register" ? "Create your local account" : "Welcome back"}
                </h2>
                <p className="auth-card-sub">
                  {needsOnboarding
                    ? "One click creates a private recovery key for this machine. No email or cloud account."
                    : "Paste the recovery key you saved for this workspace to unlock it."}
                </p>
              </div>

              <form onSubmit={submit} className="auth-form">
                {mode === "register" ? (
                  <>
                    {storagePicker}
                    {error && (
                      <p className="auth-error" role="alert">
                        {error}
                      </p>
                    )}
                    <div className="auth-primary-actions">
                      <button
                        type="button"
                        className="btn btn-primary auth-submit"
                        disabled={busy}
                        onClick={() => void generateAndContinue()}
                      >
                        {busy ? "Working…" : "Create account and continue"}
                      </button>
                    </div>
                    <details className="auth-tech-details">
                      <summary>Already have a recovery key?</summary>
                      <div className="auth-restore-block">
                        {recoveryKeyField}
                        <button
                          type="submit"
                          className="btn auth-submit"
                          disabled={
                            busy || !nsec.trim() || (needsImportPassphrase && !importPassphrase)
                          }
                        >
                          Restore and continue
                        </button>
                      </div>
                    </details>
                  </>
                ) : (
                  <>
                    {recoveryKeyField}
                    {error && (
                      <p className="auth-error" role="alert">
                        {error}
                      </p>
                    )}
                    <button
                      type="submit"
                      className="btn btn-primary auth-submit"
                      disabled={
                        busy || !nsec.trim() || (needsImportPassphrase && !importPassphrase)
                      }
                    >
                      {busy ? "Signing in…" : "Sign in"}
                    </button>
                    <button
                      type="button"
                      className="auth-reset-link"
                      onClick={resetForgottenKey}
                      disabled={resetting}
                    >
                      {resetting
                        ? "Starting a new session..."
                        : "Lost your recovery key? Start a new session"}
                    </button>
                  </>
                )}
              </form>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
