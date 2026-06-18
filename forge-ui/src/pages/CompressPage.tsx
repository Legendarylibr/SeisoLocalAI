import { useEffect, useState } from "react";
import { api, CompressJob, CompressPreset, TrainableModel } from "@/lib/api";
import { PipelineJobPanel } from "@/components/studio/PipelineJobPanel";
import { StagePipelineJobsTable } from "@/components/studio/StagePipelineJobsTable";
import { StudioPageShell } from "@/components/StudioPageShell";
import { HfBaseModelPicker } from "@/components/HfBaseModelPicker";
import { usePipelineJobStream } from "@/hooks/usePipelineJobStream";
import { useStagePipelinePresets } from "@/hooks/useStagePipelinePresets";
import { resolveModelChoice, writeStoredModel } from "@/lib/modelSelection";

const FALLBACK_PRESETS: CompressPreset[] = [
  { id: "smoke", label: "Smoke", stages: ["distill", "prune", "finetune", "evaluate", "export"] },
  { id: "full", label: "Full", stages: ["distill", "prune", "finetune", "evaluate", "export"] },
  { id: "distill_only", label: "Distill Only", stages: ["distill", "evaluate"] },
  { id: "prune_recover", label: "Prune Recover", stages: ["prune", "finetune", "evaluate", "export"] },
  { id: "quantize", label: "Quantize", stages: ["quantize_gptq", "evaluate", "export"] },
];

const FALLBACK_STAGES = [
  ...new Set([
    ...FALLBACK_PRESETS.flatMap((p) => p.stages),
    "quantize_gptq",
    "quantize_awq",
  ]),
];

export function CompressPage() {
  const [jobs, setJobs] = useState<CompressJob[]>([]);
  const {
    preset,
    setPreset,
    presetList,
    allStages,
    stageHelp,
    selectedStages,
    toggleStage,
  } = useStagePipelinePresets(FALLBACK_PRESETS, FALLBACK_STAGES, api.compressPresets);
  const [localModels, setLocalModels] = useState<TrainableModel[]>([]);
  const [teacherModel, setTeacherModel] = useState("");
  const [studentModel, setStudentModel] = useState("");
  const [modelsReady, setModelsReady] = useState(false);
  const [modelDir, setModelDir] = useState("");
  const [distillSteps, setDistillSteps] = useState<number | "">("");
  const [finetuneSteps, setFinetuneSteps] = useState<number | "">("");
  const [pruneRatio, setPruneRatio] = useState(0.25);
  const [pruneMethod, setPruneMethod] = useState("magnitude");
  const [maxTrainSamples, setMaxTrainSamples] = useState<number | "">("");
  const [calibrationSamples, setCalibrationSamples] = useState<number | "">("");
  const [exportModelName, setExportModelName] = useState("seiso-compressed");
  const [seed, setSeed] = useState(42);
  const [deterministic, setDeterministic] = useState(true);
  const [configFile, setConfigFile] = useState("");
  const [linkTrainingJob, setLinkTrainingJob] = useState("");
  const { logs, result, activeJob, resetStream, watchJob } = usePipelineJobStream();
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    api.listCompressJobs().then(setJobs).catch(console.error);
  }, []);

  useEffect(() => {
    let cancelled = false;
    Promise.all([api.compressPresets(), api.listTrainingModels()])
      .then(([presetsResp, localResp]) => {
        if (cancelled) return;
        setLocalModels(localResp.models);
        const defaults = presetsResp.defaults ?? {};
        const localRepos = localResp.models
          .map((m) => m.repo_id)
          .filter((repo): repo is string => !!repo);
        setTeacherModel(
          resolveModelChoice("compress:teacher", defaults.teacher_model, localRepos),
        );
        setStudentModel(
          resolveModelChoice("compress:student", defaults.student_model, localRepos),
        );
        setModelsReady(true);
      })
      .catch(console.error);
    return () => {
      cancelled = true;
    };
  }, []);

  const refreshJobs = () => api.listCompressJobs().then(setJobs).catch(console.error);

  const start = async () => {
    setStarting(true);
    resetStream();
    try {
      const body: Record<string, unknown> = {
        preset,
        teacher_model: teacherModel,
        student_model: studentModel,
        prune_ratio: pruneRatio,
        prune_method: pruneMethod,
        seed,
        deterministic,
        export_model_name: exportModelName,
      };
      if (selectedStages.length) body.stages = selectedStages;
      if (modelDir) body.model_dir = modelDir;
      if (configFile) body.config_file = configFile;
      if (distillSteps !== "") body.distill_steps = distillSteps;
      if (finetuneSteps !== "") body.finetune_steps = finetuneSteps;
      if (maxTrainSamples !== "") body.max_train_samples = maxTrainSamples;
      if (calibrationSamples !== "") body.calibration_samples = calibrationSamples;
      if (linkTrainingJob) body.link_training_job_id = linkTrainingJob;

      const res = await api.startCompress(body);
      watchJob(`/compress/jobs/${res.job_id}/stream`, res.job_id, {
        onResult: () => refreshJobs(),
      });
      refreshJobs();
    } finally {
      setStarting(false);
    }
  };

  return (
    <StudioPageShell
      title="Model Compression"
      subtitle="Code Llama compression pipeline: distillation, MLP pruning, recovery fine-tune, evaluation, and export bundles (vLLM/Docker/GGUF scripts). Hash-chained manifests for reproducibility."
    >
      <div className="train-layout">
        <div className="card compress-config-card studio-card">
          <div className="studio-card-head">
            <span className="studio-card-icon" aria-hidden>⚙</span>
            <div className="studio-card-head-text">
              <div className="studio-card-title">Pipeline configuration</div>
              <div className="studio-card-desc">Presets, stages, models, and training parameters</div>
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

          <h3 className="section-title">Models</h3>
          <label>Teacher model</label>
          <HfBaseModelPicker
            value={teacherModel}
            localModels={localModels}
            disabled={!modelsReady}
            onChange={(value) => {
              setTeacherModel(value);
              writeStoredModel("compress:teacher", value);
            }}
          />

          <label>Student model</label>
          <HfBaseModelPicker
            value={studentModel}
            localModels={localModels}
            disabled={!modelsReady}
            onChange={(value) => {
              setStudentModel(value);
              writeStoredModel("compress:student", value);
            }}
          />

          <label>Starting model dir (optional — for prune/finetune presets)</label>
          <input value={modelDir} onChange={(e) => setModelDir(e.target.value)} placeholder="~/.seiso/checkpoints/…" />

          <label>Link training job ID (optional)</label>
          <input value={linkTrainingJob} onChange={(e) => setLinkTrainingJob(e.target.value)} placeholder="uuid from Train page" />

          <h3 className="section-title">Training & pruning</h3>
          <div className="option-grid">
            <div>
              <label>Distill steps</label>
              <input
                type="number"
                min={1}
                value={distillSteps}
                onChange={(e) => setDistillSteps(e.target.value ? +e.target.value : "")}
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

          <div className="option-grid">
            <div>
              <label>Prune ratio: {pruneRatio.toFixed(2)}</label>
              <input
                type="range"
                min={0.05}
                max={0.5}
                step={0.05}
                value={pruneRatio}
                onChange={(e) => setPruneRatio(+e.target.value)}
              />
            </div>
            <div>
              <label>Prune method</label>
              <select value={pruneMethod} onChange={(e) => setPruneMethod(e.target.value)}>
                <option value="magnitude">Magnitude</option>
                <option value="wanda">Wanda</option>
              </select>
            </div>
          </div>

          <label>Max train samples (override)</label>
          <input
            type="number"
            min={1}
            value={maxTrainSamples}
            onChange={(e) => setMaxTrainSamples(e.target.value ? +e.target.value : "")}
            placeholder="preset default"
          />

          <h3 className="section-title">Export & quantization</h3>
          <label>Export model name</label>
          <input value={exportModelName} onChange={(e) => setExportModelName(e.target.value)} />

          <label>Calibration samples (GPTQ / AWQ)</label>
          <input
            type="number"
            min={1}
            value={calibrationSamples}
            onChange={(e) => setCalibrationSamples(e.target.value ? +e.target.value : "")}
            placeholder="preset default"
          />

          <h3 className="section-title">Reproducibility</h3>
          <div className="option-grid">
            <div>
              <label>Seed</label>
              <input
                type="number"
                min={0}
                value={seed}
                onChange={(e) => setSeed(+e.target.value)}
              />
            </div>
            <div className="checkbox-group" style={{ margin: 0, justifyContent: "flex-end" }}>
              <label>
                <input
                  type="checkbox"
                  checked={deterministic}
                  onChange={(e) => setDeterministic(e.target.checked)}
                />
                Deterministic mode
              </label>
            </div>
          </div>

          <details className="config-advanced">
            <summary>Advanced options</summary>
            <label>Config file path (optional JSON override)</label>
            <input
              value={configFile}
              onChange={(e) => setConfigFile(e.target.value)}
              placeholder="~/.seiso/configs/compress.json"
            />
          </details>

          <button className="btn btn-primary btn-lg studio-action-bar-standalone" onClick={start} disabled={starting || !modelsReady || !teacherModel || !studentModel}>
            {starting ? "Starting…" : "Run compression pipeline"}
          </button>
        </div>

        <PipelineJobPanel activeJob={activeJob} logs={logs} result={result} />
      </div>

      <StagePipelineJobsTable jobs={jobs} emptyMessage="No compression jobs yet." />
    </StudioPageShell>
  );
}
