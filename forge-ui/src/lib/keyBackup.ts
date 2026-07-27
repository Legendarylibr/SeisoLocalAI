/** One-time write-down payload after Forge generates a Nostr key. */

import {
  encryptNip49,
  nsecToSecretBytes,
  type KeySecurityByte,
} from "./nip49";

export type KeyBackup = {
  /** Private key — required to sign in later. */
  nsec: string;
  /** Public identity (safe to share). */
  npub: string;
};

export const KEY_BACKUP_STORAGE = "seiso_key_backup";

export function parseKeyBackup(raw: string | null | undefined): KeyBackup | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as Partial<KeyBackup>;
    const nsec = typeof parsed?.nsec === "string" ? parsed.nsec.trim() : "";
    const npub = typeof parsed?.npub === "string" ? parsed.npub.trim() : "";
    if (!nsec.startsWith("nsec1") || !npub.startsWith("npub1")) return null;
    return { nsec, npub };
  } catch {
    return null;
  }
}

export function readStoredKeyBackup(
  storage: Pick<Storage, "getItem"> = sessionStorage,
): KeyBackup | null {
  try {
    return parseKeyBackup(storage.getItem(KEY_BACKUP_STORAGE));
  } catch {
    return null;
  }
}

export function persistKeyBackup(
  backup: KeyBackup | null,
  storage: Pick<Storage, "setItem" | "removeItem"> = sessionStorage,
): void {
  try {
    if (backup?.nsec && backup?.npub) {
      storage.setItem(KEY_BACKUP_STORAGE, JSON.stringify(backup));
    } else {
      storage.removeItem(KEY_BACKUP_STORAGE);
    }
  } catch {
    /* ignore quota / private mode */
  }
}

export const KEY_BACKUP_FILENAME = "seiso-ncryptsec-backup.txt";

/** NIP-49 encrypted backup file contents (no raw nsec). */
export function formatKeyBackupTxt(ncryptsec: string, npub: string): string {
  return [
    "# Seiso Local AI — NIP-49 encrypted key backup (ncryptsec)",
    "# This file does NOT contain your raw nsec.",
    "# Decrypt with your passphrase to sign in, or paste ncryptsec + passphrase in Forge.",
    "# Keep the passphrase separate from this file.",
    "",
    `ncryptsec=${ncryptsec}`,
    `npub=${npub}`,
    "",
  ].join("\n");
}

export type DownloadNip49Options = {
  passphrase: string;
  /** Default 16 (~64 MiB / ~100ms). */
  logn?: number;
  /**
   * 0x00 = known insecure handling (e.g. shown on screen).
   * Onboarding write-down always shows nsec once, so default is 0x00.
   */
  keySecurity?: KeySecurityByte;
  filename?: string;
};

/** Encrypt with NIP-49 and trigger a same-tab .txt download (no raw nsec in file). */
export async function downloadNip49KeyBackup(
  backup: KeyBackup,
  options: DownloadNip49Options,
): Promise<void> {
  const passphrase = options.passphrase;
  if (!passphrase || passphrase.length < 8) {
    throw new Error("Passphrase must be at least 8 characters");
  }
  const secret = nsecToSecretBytes(backup.nsec);
  const ncryptsec = encryptNip49(
    secret,
    passphrase,
    options.logn ?? 16,
    options.keySecurity ?? 0x00,
  );
  const blob = new Blob([formatKeyBackupTxt(ncryptsec, backup.npub)], {
    type: "text/plain;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = options.filename ?? KEY_BACKUP_FILENAME;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
