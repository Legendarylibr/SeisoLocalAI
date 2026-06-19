const PREFIX = "seiso:model:";

export function readStoredModel(key: string): string | null {
  try {
    const value = localStorage.getItem(`${PREFIX}${key}`);
    return value?.trim() ? value : null;
  } catch {
    return null;
  }
}

export function writeStoredModel(key: string, value: string): void {
  try {
    const trimmed = value.trim();
    if (!trimmed) {
      localStorage.removeItem(`${PREFIX}${key}`);
      return;
    }
    localStorage.setItem(`${PREFIX}${key}`, trimmed);
  } catch {
    /* ignore quota / private mode */
  }
}

/** Prefer saved choice, then API default, then first locally cached repo. */
export function resolveModelChoice(
  storageKey: string,
  apiDefault: string | undefined,
  localRepos: string[],
): string {
  const stored = readStoredModel(storageKey);
  if (stored) return stored;
  if (apiDefault?.trim()) return apiDefault;
  return localRepos[0] ?? "";
}
