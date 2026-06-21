/** Rough output token estimate — matches server benchmark heuristic. */
export function estimateOutputTokens(text: string): number {
  const stripped = text.trim();
  if (!stripped) return 0;
  return Math.max(1, Math.round(stripped.split(/\s+/).length * 1.35));
}

export function computeTokensPerSec(tokenCount: number, elapsedMs: number): number | null {
  if (tokenCount <= 0 || elapsedMs <= 0) return null;
  return tokenCount / (elapsedMs / 1000);
}

export function formatTokensPerSec(tps: number): string {
  if (!Number.isFinite(tps) || tps <= 0) return "—";
  if (tps >= 100) return `${Math.round(tps)} tok/s`;
  return `${tps.toFixed(1)} tok/s`;
}
