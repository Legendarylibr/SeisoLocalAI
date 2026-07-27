/** One-time write-down payload after Forge generates a Nostr key. */

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

export const KEY_BACKUP_FILENAME = "seiso-nsec-backup.txt";

/** Plain-text contents for a same-window .txt download. */
export function formatKeyBackupTxt(backup: KeyBackup): string {
  return [
    "# Seiso Local AI — Nostr key backup",
    "# Keep this file secret and offline.",
    "# Anyone with the nsec can unlock this Seiso instance.",
    "# Sign in later by pasting the nsec. The npub is public identity only.",
    "",
    `nsec=${backup.nsec}`,
    `npub=${backup.npub}`,
    "",
  ].join("\n");
}

/** Trigger a same-tab download of the key backup as a .txt file. */
export function downloadKeyBackupTxt(
  backup: KeyBackup,
  filename: string = KEY_BACKUP_FILENAME,
): void {
  const blob = new Blob([formatKeyBackupTxt(backup)], {
    type: "text/plain;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
