export type StreamStats = {
  output_tokens: number;
  finish_reason?: string | null;
  truncated?: boolean;
  auto_continues?: number;
};

/** Parse server-reported stream stats (falls back to null on invalid payloads). */
export function parseStreamStats(data: string): StreamStats | null {
  try {
    const parsed = JSON.parse(data) as {
      output_tokens?: unknown;
      finish_reason?: unknown;
      truncated?: unknown;
      auto_continues?: unknown;
    };
    const outputTokens = parsed.output_tokens;
    if (typeof outputTokens !== "number" || !Number.isFinite(outputTokens) || outputTokens < 0) {
      return null;
    }
    const stats: StreamStats = { output_tokens: Math.floor(outputTokens) };
    if (typeof parsed.finish_reason === "string" && parsed.finish_reason) {
      stats.finish_reason = parsed.finish_reason;
    }
    if (typeof parsed.truncated === "boolean") {
      stats.truncated = parsed.truncated;
    }
    if (
      typeof parsed.auto_continues === "number" &&
      Number.isFinite(parsed.auto_continues) &&
      parsed.auto_continues >= 0
    ) {
      stats.auto_continues = Math.floor(parsed.auto_continues);
    }
    return stats;
  } catch {
    return null;
  }
}

/** Rough output token estimate when the server does not stream token counts. */
export function estimateOutputTokens(text: string): number {
  const stripped = text.trim();
  if (!stripped) return 0;
  return Math.max(1, Math.round(stripped.split(/\s+/).length * 1.35));
}

export function resolveOutputTokenCount(
  measuredTokens: number,
  fallbackText: string,
): number {
  if (measuredTokens > 0) return measuredTokens;
  return estimateOutputTokens(fallbackText);
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
