import { useEffect, useState } from "react";
import { api, ImageCompressJob, ImageCompressPreset } from "@/lib/api";
import { StagePipelineShell } from "@/components/studio/StagePipelineShell";
import { HfBaseModelPicker } from "@/components/HfBaseModelPicker";
import { useStagePipelinePage } from "@/hooks/useStagePipelinePage";
import { resolveModelChoice, writeStoredModel } from "@/lib/modelSelection";

const IMAGE_STAGE_ORDER = [
  "baseline",
  "distill_progressive",
  "distill_clip",
  "distill_cfg",
  "evaluate_distilled",
  "prune",
  "evaluate_pruned",
  "finetune",
  "evaluate_finetuned",
  "quantize",
  "evaluate_quantized",
  "optimize",
  "export_onnx",
  "export_shard",
  "lora_test",
  "report",
];

const FALLBACK_PRESETS: ImageCompressPreset[] = [
  {
    id: "smoke",
    label: "Smoke",
    stages: ["baseline", "distill_progressive", "prune", "quantize", "evaluate_quantized", "report"],
  },
  { id: "full", label: "Full", stages: IMAGE_STAGE_ORDER },
  {
    id: "distill_only",
    label: "Distill Only",
    stages: ["baseline", "distill_progressive", "distill_clip", "distill_cfg", "evaluate_distilled", "report"],
  },
  {
    id: "prune_recover",
    label: "Prune Recover",
    stages: ["prune", "evaluate_pruned", "finetune", "evaluate_finetuned", "report"],
  },
  {
    id: "quantize",
    label: "Quantize",
    stages: ["quantize", "evaluate_quantized", "export_shard", "report"],
  },
];

const FALLBACK_STAGES = [...new Set([...IMAGE_STAGE_ORDER, ...FALLBACK_PRESETS.flatMap((p) => p.stages)])];

const IMAGE_COMPRESS_PIPELINE = {
  fallbackPresets: FALLBACK_PRESETS,
  fallbackStages: FALLBACK_STAGES,
  loadPresets: api.imageCompressPresets,
  listJobs: api.listImageCompressJobs,
  startJob: api.startImageCompress,
  streamPath: (jobId: string) => `/image-compress/jobs/${jobId}/stream`,
};

export function ImageCompressPage() {
  const {
    jobs,
    localModels,
    modelsReady,
    starting,
    runPipeline,
    logs,
    result,
    activeJob,
    preset,
    setPreset,
    presetList,
    allStages,
    stageHelp,
    selectedStages,
    toggleStage,
    defaults,
  } = useStagePipelinePage<ImageCompressJob>(IMAGE_COMPRESS_PIPELINE);

  const [baseModel, setBaseModel] = useState("");
  const [modelDir, setModelDir] = useState("");
  const [dataPath, setDataPath] = useState("");
  const [steps, setSteps] = useState<number | "">("");
  const [clipDistillSteps, setClipDistillSteps] = useState<number | "">("");
  const [cfgDistillSteps, setCfgDistillSteps] = useState<number | "">("");
  const [finetuneSteps, setFinetuneSteps] = useState<number | "">("");
  const [pruneRatio, setPruneRatio] = useState(0.3);
  const [textEncoderPruneRatio, setTextEncoderPruneRatio] = useState<number | "">("");
  const [vaePruneRatio, setVaePruneRatio] = useState<number | "">("");
  const [int8CalibrationSamples, setInt8CalibrationSamples] = useState<number | "">("");
  const [evalSamples, setEvalSamples] = useState<number | "">("");
  const [evalInferenceSteps, setEvalInferenceSteps] = useState<number | "">("");
  const [exportModelName, setExportModelName] = useState("seiso-sd-compressed");

  useEffect(() => {
    if (!modelsReady) return;
    const localRepos = localModels
      .map((m) => m.repo_id)
      .filter((repo): repo is string => !!repo);
    setBaseModel(resolveModelChoice("image-compress:base", defaults.base_model, localRepos));
  }, [modelsReady, localModels, defaults.base_model]);

  const start = async () => {
    const body: Record<string, unknown> = {
      preset,
      base_model: baseModel,
      prune_ratio: pruneRatio,
      export_model_name: exportModelName,
    };
    if (selectedStages.length) body.stages = selectedStages;
    if (modelDir) body.model_dir = modelDir;
    if (dataPath) body.data_path = dataPath;
    if (steps !== "") body.steps = steps;
    if (clipDistillSteps !== "") body.clip_distill_steps = clipDistillSteps;
    if (cfgDistillSteps !== "") body.cfg_distill_steps = cfgDistillSteps;
    if (finetuneSteps !== "") body.finetune_steps = finetuneSteps;
    if (textEncoderPruneRatio !== "") body.text_encoder_prune_ratio = textEncoderPruneRatio;
    if (vaePruneRatio !== "") body.vae_prune_ratio = vaePruneRatio;
    if (int8CalibrationSamples !== "") body.int8_calibration_samples = int8CalibrationSamples;
    if (evalSamples !== "") body.eval_samples = evalSamples;
    if (evalInferenceSteps !== "") body.eval_inference_steps = evalInferenceSteps;
    await runPipeline(body);
  };

  return (
    <StagePipelineShell
      title="Image Compression"
      subtitle="Stable Diffusion pipeline: progressive distillation, structured pruning, INT8 quantisation, and deployment exports (ONNX, sharded safetensors)."
      cardIcon="🖼"
      cardDesc="Presets, stages, base model, and tuning parameters"
      preset={preset}
      setPreset={setPreset}
      presetList={presetList}
      allStages={allStages}
      stageHelp={stageHelp}
      selectedStages={selectedStages}
      toggleStage={toggleStage}
      logs={logs}
      result={result}
      activeJob={activeJob}
      jobs={jobs}
      jobsEmptyMessage="No image compression jobs yet."
      canStart={modelsReady && !!baseModel}
      starting={starting}
      onStart={start}
      startLabel="Run image compression pipeline"
    >
      <h3 className="section-title">Model & data</h3>
      <label>Base model</label>
      <HfBaseModelPicker
        value={baseModel}
        localModels={localModels}
        disabled={!modelsReady}
        onChange={(value) => {
          setBaseModel(value);
          writeStoredModel("image-compress:base", value);
        }}
      />

      <label>Model dir (optional — starting checkpoint)</label>
      <input value={modelDir} onChange={(e) => setModelDir(e.target.value)} placeholder="~/.seiso/models/…" />

      <label>Calibration data path (optional)</label>
      <input value={dataPath} onChange={(e) => setDataPath(e.target.value)} placeholder="~/.seiso/data/…" />

      <h3 className="section-title">Distillation & fine-tune</h3>
      <div className="option-grid">
        <div>
          <label>UNet distill steps</label>
          <input
            type="number"
            min={1}
            value={steps}
            onChange={(e) => setSteps(e.target.value ? +e.target.value : "")}
            placeholder="preset default"
          />
        </div>
        <div>
          <label>CLIP distill steps</label>
          <input
            type="number"
            min={1}
            value={clipDistillSteps}
            onChange={(e) => setClipDistillSteps(e.target.value ? +e.target.value : "")}
            placeholder="preset default"
          />
        </div>
      </div>
      <div className="option-grid">
        <div>
          <label>CFG distill steps</label>
          <input
            type="number"
            min={1}
            value={cfgDistillSteps}
            onChange={(e) => setCfgDistillSteps(e.target.value ? +e.target.value : "")}
            placeholder="preset default"
          />
        </div>
        <div>
          <label>Finetune steps</label>
          <input
            type="number"
            min={1}
            value={finetuneSteps}
            onChange={(e) => setFinetuneSteps(e.target.value ? +e.target.value : "")}
            placeholder="preset default"
          />
        </div>
      </div>

      <h3 className="section-title">Pruning</h3>
      <label>UNet prune ratio: {pruneRatio.toFixed(2)}</label>
      <input
        type="range"
        min={0.05}
        max={0.6}
        step={0.05}
        value={pruneRatio}
        onChange={(e) => setPruneRatio(+e.target.value)}
      />
      <div className="option-grid">
        <div>
          <label>Text encoder prune ratio</label>
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={textEncoderPruneRatio}
            onChange={(e) => setTextEncoderPruneRatio(e.target.value ? +e.target.value : "")}
            placeholder="preset default"
          />
        </div>
        <div>
          <label>VAE prune ratio</label>
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={vaePruneRatio}
            onChange={(e) => setVaePruneRatio(e.target.value ? +e.target.value : "")}
            placeholder="preset default"
          />
        </div>
      </div>

      <h3 className="section-title">Evaluation & export</h3>
      <div className="option-grid">
        <div>
          <label>Eval samples</label>
          <input
            type="number"
            min={1}
            value={evalSamples}
            onChange={(e) => setEvalSamples(e.target.value ? +e.target.value : "")}
            placeholder="preset default"
          />
        </div>
        <div>
          <label>Eval inference steps</label>
          <input
            type="number"
            min={1}
            value={evalInferenceSteps}
            onChange={(e) => setEvalInferenceSteps(e.target.value ? +e.target.value : "")}
            placeholder="preset default"
          />
        </div>
      </div>
      <label>INT8 calibration samples</label>
      <input
        type="number"
        min={1}
        value={int8CalibrationSamples}
        onChange={(e) => setInt8CalibrationSamples(e.target.value ? +e.target.value : "")}
        placeholder="preset default"
      />
      <label>Export model name</label>
      <input value={exportModelName} onChange={(e) => setExportModelName(e.target.value)} />
    </StagePipelineShell>
  );
}
