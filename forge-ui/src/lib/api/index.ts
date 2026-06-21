import { authApi } from "./auth";
import { clearLegacyToken } from "./client";
import { compressApi } from "./compress";
import { distillRlApi } from "./distillRl";
import { exportApi } from "./export";
import { inferenceApi } from "./inference";
import { knowledgeApi } from "./knowledge";
import { modelsApi } from "./models";
import { providersApi } from "./providers";
import { recipesApi } from "./recipes";
import { rlQuantApi } from "./rlQuant";
import { settingsApi } from "./settings";
import { streamChat, subscribeSSE } from "./sse";
import { systemApi } from "./system";
import { trainingApi } from "./training";

export const api = {
  ...authApi,
  ...modelsApi,
  ...exportApi,
  ...inferenceApi,
  ...trainingApi,
  ...rlQuantApi,
  ...compressApi,
  ...distillRlApi,
  ...settingsApi,
  ...recipesApi,
  ...providersApi,
  ...systemApi,
  ...knowledgeApi,
};

export { clearLegacyToken, streamChat, subscribeSSE };
export type * from "./types";
