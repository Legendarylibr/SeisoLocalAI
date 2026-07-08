export const CHAT_INFERENCE_SETTINGS_KEY = "seiso.chat.inference_settings";
export const CHAT_MAX_TOKENS_BY_MODEL_KEY = "seiso.chat.max_tokens_by_model";

export type ChatInferenceSettings = {
  temperature: number;
  topPEnabled: boolean;
  topP: number;
  specEnabled: boolean;
  draftModelId: string;
  numSpeculativeTokens: number;
  maxTokens: number;
  panelOpen: boolean;
};

const DEFAULTS: ChatInferenceSettings = {
  temperature: 0.7,
  topPEnabled: false,
  topP: 0.9,
  specEnabled: false,
  draftModelId: "",
  numSpeculativeTokens: 4,
  maxTokens: 2048,
  panelOpen: false,
};

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

function readMaxTokensByModelMap(): Record<string, number> {
  try {
    const raw = localStorage.getItem(CHAT_MAX_TOKENS_BY_MODEL_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const out: Record<string, number> = {};
    for (const [key, value] of Object.entries(parsed)) {
      const num = clamp(Math.round(Number(value)), 1, 8192);
      if (Number.isFinite(num)) out[key] = num;
    }
    return out;
  } catch {
    return {};
  }
}

export function readStoredMaxTokensForModel(
  modelId: string | null | undefined,
): number | null {
  if (!modelId) return null;
  const value = readMaxTokensByModelMap()[modelId];
  return typeof value === "number" ? value : null;
}

export function writeStoredMaxTokensForModel(
  modelId: string | null | undefined,
  value: number,
): void {
  if (!modelId) return;
  try {
    const map = readMaxTokensByModelMap();
    map[modelId] = clamp(Math.round(Number(value)), 1, 8192);
    localStorage.setItem(CHAT_MAX_TOKENS_BY_MODEL_KEY, JSON.stringify(map));
  } catch {
    /* ignore */
  }
}

export function readChatInferenceSettings(): ChatInferenceSettings {
  try {
    const raw = localStorage.getItem(CHAT_INFERENCE_SETTINGS_KEY);
    if (!raw) return { ...DEFAULTS };
    const parsed = JSON.parse(raw) as Partial<ChatInferenceSettings>;
    return {
      temperature: clamp(Number(parsed.temperature ?? DEFAULTS.temperature), 0, 2),
      topPEnabled: Boolean(parsed.topPEnabled),
      topP: clamp(Number(parsed.topP ?? DEFAULTS.topP), 0.05, 1),
      specEnabled: Boolean(parsed.specEnabled),
      draftModelId: typeof parsed.draftModelId === "string" ? parsed.draftModelId : "",
      numSpeculativeTokens: clamp(
        Math.round(Number(parsed.numSpeculativeTokens ?? DEFAULTS.numSpeculativeTokens)),
        1,
        32,
      ),
      maxTokens: clamp(Math.round(Number(parsed.maxTokens ?? DEFAULTS.maxTokens)), 512, 8192),
      panelOpen: Boolean(parsed.panelOpen),
    };
  } catch {
    return { ...DEFAULTS };
  }
}

export function writeChatInferenceSettings(partial: Partial<ChatInferenceSettings>): void {
  try {
    const next = { ...readChatInferenceSettings(), ...partial };
    localStorage.setItem(CHAT_INFERENCE_SETTINGS_KEY, JSON.stringify(next));
  } catch {
    /* ignore */
  }
}
