/** Pure helpers for TrainPage recommendation auto-apply behavior. */

export type TrainRecConfig = {
  dataset_format?: string;
  train_on_responses_only?: boolean;
};

export function shouldAutoApplyRecommendationFields(
  configCustomized: boolean,
  recConfig: TrainRecConfig | undefined,
  datasetAnalysis: unknown,
): boolean {
  return !configCustomized && !!recConfig && !datasetAnalysis;
}

export function pickAutoRecommendationFields(recConfig: TrainRecConfig): {
  datasetFormat?: string;
  trainOnResponsesOnly?: boolean;
} {
  const out: { datasetFormat?: string; trainOnResponsesOnly?: boolean } = {};
  if (recConfig.dataset_format && recConfig.dataset_format !== "auto") {
    out.datasetFormat = recConfig.dataset_format;
  }
  if (recConfig.train_on_responses_only != null) {
    out.trainOnResponsesOnly = recConfig.train_on_responses_only;
  }
  return out;
}
