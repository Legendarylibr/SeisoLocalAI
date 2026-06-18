import { useEffect, useState } from "react";
import { api, CompressJob, CompressPreset } from "@/lib/api";
import { StagePipelineShell } from "@/components/studio/StagePipelineShell";
import { HfBaseModelPicker } from "@/components/HfBaseModelPicker";
import { useStagePipelinePage } from "@/hooks/useStagePipelinePage";
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

const COMPRESS_PIPELINE = {
  fallbackPresets: FALLBACK_PRESETS,
  fallbackStages: FALLBACK_STAGES,
  loadPresets: api.compressPresets,
  listJobs: api.listCompressJobs,
  startJob: api.startCompress,
  streamPath: (jobId: string) => `/compress/jobs/${jobId}/stream`,
};

export function CompressPage() {
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
  } = useStagePipelinePage<CompressJob>(COMPRESS_PIPELINE);

  const [teacherModel, setTeacherModel] = useState("");
  const [studentModel, setStudentModel] = useState("");
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

  useEffect(() => {
    if (!modelsReady) return;
    const localRepos = localModels
      .map((m) => m.repo_id)
      .filter((repo): repo is string => !!repo);
    setTeacherModel(resolveModelChoice("compress:teacher", defaults.teacher_model, localRepos));
    setStudentModel(resolveModelChoice("compress:student", defaults.student_model, localRepos));
  }, [modelsReady, localModels, defaults.teacher_model, defaults.student_model]);

  const start = async () => {
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
    await runPipeline(body);
  };

  return (
    <StagePipelineShell
      title="Model Compression"
      subtitle="Code Llama compression pipeline: distillation, MLP pruning, recovery fine-tune, evaluation, and export bundles (vLLM/Docker/GGUF scripts). Hash-chained manifests for reproducibility."
      cardIcon="⚙"
      cardDesc="Presets, stages, models, and training parameters"
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
      jobsEmptyMessage="No compression jobs yet."
      canStart={modelsReady && !!teacherModel && !!studentModel}
      starting={starting}
      onStart={start}
      startLabel="Run compression pipeline"
    >
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
          <input type="number" min={0} value={seed} onChange={(e) => setSeed(+e.target.value)} />
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
    </StagePipelineShell>
  );
}
