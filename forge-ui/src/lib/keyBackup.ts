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
