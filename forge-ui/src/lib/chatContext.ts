import type { ChatContextStatus } from "@/lib/api/types";

export const CHAT_CTX_STORAGE_KEY = "seiso.chat.context_window";

export type ContextWindowSetting = "auto" | 2048 | 4096 | 8192;

export function readStoredContextWindow(): ContextWindowSetting {
  try {
    const raw = localStorage.getItem(CHAT_CTX_STORAGE_KEY);
    if (raw === "auto" || raw === "2048" || raw === "4096" || raw === "8192") {
      return raw === "auto" ? "auto" : Number(raw) as 2048 | 4096 | 8192;
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
  const msgs =
    status.messages_omitted > 0
      ? `${status.messages_included}/${status.message_count} msgs`
      : `${status.message_count} msgs`;
  return `${used} / ${limit} ctx · ${msgs}`;
}
