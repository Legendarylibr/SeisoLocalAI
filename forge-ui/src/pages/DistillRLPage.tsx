import { useEffect, useState } from "react";
import { api, DistillRLJob } from "@/lib/api";
import { FormSection } from "@/components/research/FormSection";
import { StagePipelineShell } from "@/components/studio/StagePipelineShell";
import { HfBaseModelPicker } from "@/components/HfBaseModelPicker";
import { useStagePipelinePage } from "@/hooks/useStagePipelinePage";
import { resolveModelChoice, writeStoredModel } from "@/lib/modelSelection";

const FALLBACK_STAGES = ["distill", "rollout", "dpo", "evaluate"];

const DISTILL_RL_PIPELINE = {
  fallbackStages: FALLBACK_STAGES,
  loadPresets: api.distillRLPresets,
  listJobs: api.listDistillRLJobs,
  startJob: api.startDistillRL,
  streamPath: (jobId: string) => `/distill-rl/jobs/${jobId}/stream`,
};

export function DistillRLPage() {
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
    presetsLoading,
    presetsReady,
    allStages,
    stageHelp,
    selectedStages,
    toggleStage,
    defaults,
  } = useStagePipelinePage<DistillRLJob>(DISTILL_RL_PIPELINE);

  const [teacherModel, setTeacherModel] = useState("");
  const [studentModel, setStudentModel] = useState("");
  const [distilledPath, setDistilledPath] = useState("");
  const [promptLibrary, setPromptLibrary] = useState("");
  const [distillSteps, setDistillSteps] = useState<number | "">("");
  const [rolloutPrompts, setRolloutPrompts] = useState<number | "">("");
  const [dpoEpochs, setDpoEpochs] = useState<number | "">("");
  const [seeds, setSeeds] = useState("");
  const [seed, setSeed] = useState(42);
  const [deterministic, setDeterministic] = useState(true);
  const [evaluateTeacher, setEvaluateTeacher] = useState(false);
  const [hashRunId, setHashRunId] = useState(false);
  const [configFile, setConfigFile] = useState("");
  const [configOverrides, setConfigOverrides] = useState("");
  const [configError, setConfigError] = useState("");
  const [linkTrainingJob, setLinkTrainingJob] = useState("");

  useEffect(() => {
    if (!modelsReady) return;
    const localRepos = localModels
      .map((m) => m.repo_id)
      .filter((repo): repo is string => !!repo);
    setTeacherModel(resolveModelChoice("distill-rl:teacher", defaults.teacher_model, localRepos));
    setStudentModel(resolveModelChoice("distill-rl:student", defaults.student_model, localRepos));
  }, [modelsReady, localModels, defaults.teacher_model, defaults.student_model]);

  const start = async () => {
    setConfigError("");
    const body: Record<string, unknown> = {
      preset,
      teacher_model: teacherModel,
      student_model: studentModel,
      seed,
      deterministic,
      evaluate_teacher: evaluateTeacher,
      hash_run_id: hashRunId,
    };
    if (selectedStages.length) body.stages = selectedStages;
    if (distilledPath) body.distilled_path = distilledPath;
    if (promptLibrary) body.prompt_library = promptLibrary;
    if (configFile) body.config_file = configFile;
    if (distillSteps !== "") body.distill_steps = distillSteps;
    if (rolloutPrompts !== "") body.rollout_max_prompts = rolloutPrompts;
    if (dpoEpochs !== "") body.dpo_epochs = dpoEpochs;
    if (linkTrainingJob) body.link_training_job_id = linkTrainingJob;
    if (seeds.trim()) {
      body.seeds = seeds
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean)
        .map((s) => Number(s));
    }
    if (configOverrides.trim()) {
      try {
        Object.assign(body, JSON.parse(configOverrides));
      } catch {
        setConfigError("Config overrides must be valid JSON.");
        return;
      }
    }
    await runPipeline(body);
  };

  return (
    <StagePipelineShell
      title="Distill-RL"
      subtitle="Teacher → student distillation, preference rollouts (teacher chosen / student rejected), and DPO fine-tuning with research artifacts and multi-seed aggregation."
      cardIcon="⚗"
      cardDesc="Presets, stages, models, and DPO parameters"
      preset={preset}
      setPreset={setPreset}
      presetList={presetList}
      presetsLoading={presetsLoading}
      allStages={allStages}
      stageHelp={stageHelp}
      selectedStages={selectedStages}
      toggleStage={toggleStage}
      logs={logs}
      result={result}
      activeJob={activeJob}
      jobs={jobs}
      jobsEmptyMessage="No distill-RL jobs yet."
      canStart={presetsReady && modelsReady && !!teacherModel && !!studentModel}
      starting={starting}
      onStart={start}
      startLabel="Run distill-RL pipeline"
    >
      <div className="studio-config-columns">
        <div className="studio-config-block">
          <FormSection title="Models" hint="Teacher generates preferred completions; student is distilled then aligned with DPO.">
            <div className="form-field">
              <label>Teacher model</label>
              <HfBaseModelPicker
                value={teacherModel}
                localModels={localModels}
                disabled={!modelsReady}
                onChange={(value) => {
                  setTeacherModel(value);
                  writeStoredModel("distill-rl:teacher", value);
                }}
              />
            </div>
            <div className="form-field">
              <label>Student model</label>
              <HfBaseModelPicker
                value={studentModel}
                localModels={localModels}
                disabled={!modelsReady}
                onChange={(value) => {
                  setStudentModel(value);
                  writeStoredModel("distill-rl:student", value);
                }}
              />
            </div>
            <div className="form-field">
              <label>Distilled checkpoint (skip distill stage)</label>
              <input
                value={distilledPath}
                onChange={(e) => setDistilledPath(e.target.value)}
                placeholder="~/.seiso/distill_rl/…/distilled"
              />
            </div>
            <div className="form-field">
              <label>Link training job ID (optional)</label>
              <input
                value={linkTrainingJob}
                onChange={(e) => setLinkTrainingJob(e.target.value)}
                placeholder="uuid from Train page"
              />
            </div>
          </FormSection>
        </div>

        <div className="studio-config-block">
          <FormSection title="Rollouts & DPO" hint="Preference dataset size and DPO training." collapsible defaultOpen={false}>
            <div className="option-grid">
              <div className="form-field">
                <label>Distill steps</label>
                <input
                  type="number"
                  min={1}
                  value={distillSteps}
                  onChange={(e) => setDistillSteps(e.target.value ? +e.target.value : "")}
                  placeholder="preset default"
                />
              </div>
              <div className="form-field">
                <label>Rollout prompts</label>
                <input
                  type="number"
                  min={1}
                  value={rolloutPrompts}
                  onChange={(e) => setRolloutPrompts(e.target.value ? +e.target.value : "")}
                  placeholder="preset default"
                />
              </div>
              <div className="form-field">
                <label>DPO epochs</label>
                <input
                  type="number"
                  min={1}
                  value={dpoEpochs}
                  onChange={(e) => setDpoEpochs(e.target.value ? +e.target.value : "")}
                  placeholder="preset default"
                />
              </div>
            </div>
            <div className="form-field">
              <label>Prompt library (JSON/JSONL)</label>
              <input
                value={promptLibrary}
                onChange={(e) => setPromptLibrary(e.target.value)}
                placeholder="Optional custom rollout prompts"
              />
            </div>
          </FormSection>
        </div>

        <div className="studio-config-block">
          <FormSection title="Reproducibility" collapsible defaultOpen={false}>
            <div className="option-grid">
              <div className="form-field">
                <label>Seed</label>
                <input type="number" min={0} value={seed} onChange={(e) => setSeed(+e.target.value)} />
              </div>
              <div className="form-field">
                <label>Multi-seed (comma-separated)</label>
                <input
                  value={seeds}
                  onChange={(e) => setSeeds(e.target.value)}
                  placeholder="13,42,99 — overrides single seed"
                />
              </div>
            </div>
            <label className="studio-checkbox-item">
              <input type="checkbox" checked={deterministic} onChange={(e) => setDeterministic(e.target.checked)} />
              Deterministic mode
            </label>
            <label className="studio-checkbox-item">
              <input
                type="checkbox"
                checked={evaluateTeacher}
                onChange={(e) => setEvaluateTeacher(e.target.checked)}
              />
              Evaluate teacher baseline
            </label>
            <label className="studio-checkbox-item">
              <input type="checkbox" checked={hashRunId} onChange={(e) => setHashRunId(e.target.checked)} />
              Hash run directory ID
            </label>
            <details className="config-advanced">
              <summary>Advanced options</summary>
              <div className="form-field">
                <label>Config file path (optional JSON override)</label>
                <input
                  value={configFile}
                  onChange={(e) => setConfigFile(e.target.value)}
                  placeholder="~/.seiso/configs/distill_rl_reproducible.json"
                />
              </div>
              <div className="form-field">
                <label>Inline config overrides (JSON)</label>
                <textarea
                  rows={5}
                  value={configOverrides}
                  onChange={(e) => setConfigOverrides(e.target.value)}
                  placeholder='{"rollout_temperature": 0.8, "dpo_beta": 0.1}'
                />
                {configError && <p className="field-error">{configError}</p>}
              </div>
            </details>
          </FormSection>
        </div>
      </div>
    </StagePipelineShell>
  );
}
