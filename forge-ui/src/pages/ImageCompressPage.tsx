import { useEffect, useState } from "react";
import { api, ImageCompressJob, ImageCompressPreset, TrainableModel } from "@/lib/api";
import { PipelineJobPanel } from "@/components/studio/PipelineJobPanel";
import { StagePipelineJobsTable } from "@/components/studio/StagePipelineJobsTable";
import { StudioPageShell } from "@/components/StudioPageShell";
import { HfBaseModelPicker } from "@/components/HfBaseModelPicker";
import { usePipelineJobStream } from "@/hooks/usePipelineJobStream";
import { useStagePipelinePresets } from "@/hooks/useStagePipelinePresets";
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

export function ImageCompressPage() {
  const [jobs, setJobs] = useState<ImageCompressJob[]>([]);
  const {
    preset,
    setPreset,
    presetList,
    allStages,
    stageHelp,
    selectedStages,
    toggleStage,
  } = useStagePipelinePresets(FALLBACK_PRESETS, FALLBACK_STAGES, api.imageCompressPresets);
  const [localModels, setLocalModels] = useState<TrainableModel[]>([]);
  const [baseModel, setBaseModel] = useState("");
  const [modelsReady, setModelsReady] = useState(false);
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
  const { logs, result, activeJob, resetStream, watchJob } = usePipelineJobStream();
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    api.listImageCompressJobs().then(setJobs).catch(console.error);
  }, []);

  useEffect(() => {
    let cancelled = false;
    Promise.all([api.imageCompressPresets(), api.listTrainingModels()])
      .then(([presetsResp, localResp]) => {
        if (cancelled) return;
        setLocalModels(localResp.models);
        const defaults = presetsResp.defaults ?? {};
        const localRepos = localResp.models
          .map((m) => m.repo_id)
          .filter((repo): repo is string => !!repo);
        setBaseModel(resolveModelChoice("image-compress:base", defaults.base_model, localRepos));
        setModelsReady(true);
      })
      .catch(console.error);
    return () => {
      cancelled = true;
    };
  }, []);

  const refreshJobs = () => api.listImageCompressJobs().then(setJobs).catch(console.error);

  const start = async () => {
    setStarting(true);
    resetStream();
    try {
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

      const res = await api.startImageCompress(body);
      watchJob(`/image-compress/jobs/${res.job_id}/stream`, res.job_id, {
        onResult: () => refreshJobs(),
      });
      refreshJobs();
    } finally {
      setStarting(false);
    }
  };

  return (
    <StudioPageShell
      title="Image Compression"
      subtitle="Stable Diffusion pipeline: progressive distillation, structured pruning, INT8 quantisation, and deployment exports (ONNX, sharded safetensors)."
    >
      <div className="train-layout">
        <div className="card compress-config-card studio-card">
          <div className="studio-card-head">
            <span className="studio-card-icon" aria-hidden>🖼</span>
            <div className="studio-card-head-text">
              <div className="studio-card-title">Pipeline configuration</div>
              <div className="studio-card-desc">Presets, stages, base model, and tuning parameters</div>
            </div>
          </div>

          <h3 className="section-title">Pipeline</h3>
          <label>Preset</label>
          <select value={preset} onChange={(e) => setPreset(e.target.value)}>
            {presetList.map((p) => (
              <option key={p.id} value={p.id}>{p.label}</option>
            ))}
          </select>

          <label>Stages</label>
          <div className="checkbox-group compress-stages">
            {allStages.map((stage) => (
              <label key={stage} title={stageHelp[stage]}>
                <input
                  type="checkbox"
                  checked={selectedStages.includes(stage)}
                  onChange={() => toggleStage(stage)}
                />
                {stage.replace(/_/g, " ")}
                {stageHelp[stage] && (
                  <span className="muted-text compress-stage-hint">{stageHelp[stage]}</span>
                )}
              </label>
            ))}
          </div>

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

          <button className="btn btn-primary btn-lg studio-action-bar-standalone" onClick={start} disabled={starting || !modelsReady || !baseModel}>
            {starting ? "Starting…" : "Run image compression pipeline"}
          </button>
        </div>

        <PipelineJobPanel activeJob={activeJob} logs={logs} result={result} />
      </div>

      <StagePipelineJobsTable jobs={jobs} emptyMessage="No image compression jobs yet." />
    </StudioPageShell>
  );
}
