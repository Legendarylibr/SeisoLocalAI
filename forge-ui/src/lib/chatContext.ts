import type { ChatContextStatus, InferenceModelOption } from "@/lib/api/types";

export const CHAT_CTX_STORAGE_KEY = "seiso.chat.context_window";
export const CHAT_CTX_BY_MODEL_KEY = "seiso.chat.context_window_by_model";

export type ContextWindowSetting = "auto" | number;

export const DEFAULT_CONTEXT_WINDOW_OPTIONS = [2048, 4096, 8192, 16384, 32768];

function readByModelMap(): Record<string, ContextWindowSetting> {
  try {
    const raw = localStorage.getItem(CHAT_CTX_BY_MODEL_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const out: Record<string, ContextWindowSetting> = {};
    for (const [key, value] of Object.entries(parsed)) {
      if (value === "auto") {
        out[key] = "auto";
        continue;
      }
      const num = Number(value);
      if (Number.isFinite(num) && num >= 2048 && num <= 131072) {
        out[key] = num;
      }
    }
    return out;
  } catch {
    return {};
  }
}

export function hasStoredContextWindowForModel(modelId: string | null | undefined): boolean {
  if (!modelId) return false;
  return Object.prototype.hasOwnProperty.call(readByModelMap(), modelId);
}

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

export function readStoredContextWindowForModel(
  modelId: string | null | undefined,
  maxAllowed = 131072,
): ContextWindowSetting {
  if (modelId) {
    const byModel = readByModelMap()[modelId];
    if (byModel === "auto") return "auto";
    if (typeof byModel === "number" && byModel >= 2048 && byModel <= maxAllowed) {
      return byModel;
    }
  }
  return readStoredContextWindow(maxAllowed);
}

export function writeStoredContextWindow(value: ContextWindowSetting): void {
  try {
    localStorage.setItem(CHAT_CTX_STORAGE_KEY, String(value));
  } catch {
    /* ignore */
  }
}

export function writeStoredContextWindowForModel(
  modelId: string | null | undefined,
  value: ContextWindowSetting,
): void {
  writeStoredContextWindow(value);
  if (!modelId) return;
  try {
    const map = readByModelMap();
    map[modelId] = value;
    localStorage.setItem(CHAT_CTX_BY_MODEL_KEY, JSON.stringify(map));
  } catch {
    /* ignore */
  }
}

export function contextWindowOptionsFromStatus(
  status: ChatContextStatus | null | undefined,
  model?: InferenceModelOption | null,
): number[] {
  const safeMax =
    typeof model?.safe_context_window_max === "number" ? model.safe_context_window_max : undefined;
  const ceiling =
    status?.n_ctx_max ||
    safeMax ||
    (typeof model?.context_ceiling === "number" ? model.context_ceiling : undefined);
  let options =
    status?.context_window_options?.filter((n) => n >= 2048) ??
    model?.safe_context_window_options?.filter((n) => n >= 2048) ??
    [];
  if (safeMax && safeMax >= 2048) {
    options = options.filter((n) => n <= safeMax);
    if (!options.includes(safeMax)) options = [...options, safeMax].sort((a, b) => a - b);
  }
  if (options.length) {
    const effectiveCeiling = safeMax || ceiling;
    if (effectiveCeiling && effectiveCeiling >= 2048 && !options.includes(effectiveCeiling)) {
      return [...options, effectiveCeiling].sort((a, b) => a - b);
    }
    return options;
  }
  if (ceiling && ceiling >= 2048) {
    return DEFAULT_CONTEXT_WINDOW_OPTIONS.filter((n) => n <= ceiling).concat(
      DEFAULT_CONTEXT_WINDOW_OPTIONS.includes(ceiling) ? [] : [ceiling],
    );
  }
  return DEFAULT_CONTEXT_WINDOW_OPTIONS;
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
