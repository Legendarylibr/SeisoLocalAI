export const CHAT_INFERENCE_SETTINGS_KEY = "seiso.chat.inference_settings";

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
