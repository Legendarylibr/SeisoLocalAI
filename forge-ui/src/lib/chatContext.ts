import type { ChatContextStatus } from "@/lib/api/types";

export const CHAT_CTX_STORAGE_KEY = "seiso.chat.context_window";

export type ContextWindowSetting = "auto" | number;

export const DEFAULT_CONTEXT_WINDOW_OPTIONS = [2048, 4096, 8192, 16384, 32768];

export function readStoredContextWindow(maxAllowed = 131072): ContextWindowSetting {
  try {
    const raw = localStorage.getItem(CHAT_CTX_STORAGE_KEY);
    if (raw === "auto") return "auto";
    const value = Number(raw);
    if (Number.isFinite(value) && value >= 2048 && value <= maxAllowed) {
      return value;
    }
  } catch {
    /* ignore */
  }
  return "auto";
}

export function writeStoredContextWindow(value: ContextWindowSetting): void {
  try {
    localStorage.setItem(CHAT_CTX_STORAGE_KEY, String(value));
  } catch {
    /* ignore */
  }
}

export function contextWindowOptionsFromStatus(
  status: ChatContextStatus | null | undefined,
): number[] {
  const options = status?.context_window_options?.filter((n) => n >= 2048) ?? [];
  return options.length ? options : DEFAULT_CONTEXT_WINDOW_OPTIONS;
}

export function normalizeContextWindow(
  value: ContextWindowSetting,
  options: number[],
): ContextWindowSetting {
  if (value === "auto") return "auto";
  if (options.includes(value)) return value;
  const max = options[options.length - 1];
  if (typeof max === "number" && value > max) return max;
  return "auto";
}

export function formatTokenCount(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0";
  if (value >= 10_000) return `${Math.round(value / 1000)}k`;
  if (value >= 1000) return `${(value / 1000).toFixed(1).replace(/\.0$/, "")}k`;
  return String(Math.round(value));
}

export function formatContextLabel(value: ContextWindowSetting): string {
  if (value === "auto") return "Auto";
  return formatTokenCount(value);
}

export function contextWindowFillClass(ratio: number): string {
  if (ratio >= 0.95) return "chat-context-fill-critical";
  if (ratio >= 0.8) return "chat-context-fill-warn";
  return "chat-context-fill-ok";
}

export function contextSummary(status: ChatContextStatus): string {
  const used = formatTokenCount(status.context_tokens_used);
  const limit = formatTokenCount(status.context_tokens_limit);
  const max = status.n_ctx_max > status.context_tokens_limit
    ? ` · max ${formatTokenCount(status.n_ctx_max)}`
    : "";
  const msgs =
    status.messages_omitted > 0
      ? `${status.messages_included}/${status.message_count} msgs`
      : `${status.message_count} msgs`;
  return `${used} / ${limit} ctx${max} · ${msgs}`;
}
