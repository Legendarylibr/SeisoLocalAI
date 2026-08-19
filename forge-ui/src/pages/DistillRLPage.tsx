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
  initialPreset: "reproducible",
};

export function DistillRLPage() {
  const {
    jobs,
    localModels,
    modelsReady,
    starting,
    startError,
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
  const [preferenceSource, setPreferenceSource] = useState("dataset");
  const [datasetRef, setDatasetRef] = useState("");
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
    // Align source defaults with presets: smoke = CI fixture; research = curated HF.
    if (preset === "smoke") {
      setPreferenceSource("grounded_library");
    } else if (preferenceSource === "grounded_library" && !promptLibrary.trim()) {
      setPreferenceSource("dataset");
    }
    // Only react to preset changes; do not fight manual source edits mid-form.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preset]);

  useEffect(() => {
    if (!modelsReady) return;
    const localRepos = localModels
      .map((m) => m.repo_id)
      .filter((repo): repo is string => !!repo);
    setTeacherModel(resolveModelChoice("distill-rl:teacher", defaults.teacher_model, localRepos));
    setStudentModel(resolveModelChoice("distill-rl:student", defaults.student_model, localRepos));
  }, [modelsReady, localModels, defaults.teacher_model, defaults.student_model]);

  const needsDatasetRef =
    preferenceSource === "dataset" && !datasetRef.trim() && !promptLibrary.trim();
  const needsPromptLibrary =
    preferenceSource === "grounded_library" && !promptLibrary.trim() && preset !== "smoke";
  const dataReady = !needsDatasetRef && !needsPromptLibrary;

  const start = async () => {
    setConfigError("");
    if (!dataReady) {
      setConfigError(
        needsDatasetRef
          ? "Set dataset_ref to a curated verifiable Hub id (or local JSONL with answer/tests)."
          : "grounded_library requires a prompt library JSON/JSONL with answer/tests.",
      );
      return;
    }
    const body: Record<string, unknown> = {
      preset,
      teacher_model: teacherModel,
      student_model: studentModel,
      preference_source: preferenceSource,
      seed,
      deterministic,
      evaluate_teacher: evaluateTeacher,
      hash_run_id: hashRunId,
    };
    if (selectedStages.length) body.stages = selectedStages;
    if (distilledPath) body.distilled_path = distilledPath;
    if (promptLibrary) body.prompt_library = promptLibrary;
    if (datasetRef.trim()) body.dataset_ref = datasetRef.trim();
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
      subtitle="Teacher → student distillation, verifiable preference rollouts (pass≻fail), and DPO with research artifacts. Smoke is CI-only; research presets need a real Hub/JSONL corpus."
      cardIcon="⚗"
      cardDesc="Presets, stages, models, and DPO parameters"
      preset={preset}
      setPreset={setPreset}
      presetList={presetList.map((p) =>
        p.id === "smoke"
          ? { ...p, label: "Smoke (CI fixture — not meaningful training)" }
          : p,
      )}
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
      canStart={
        presetsReady && modelsReady && !!teacherModel && !!studentModel && dataReady
      }
      starting={starting}
      startError={startError}
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
          <FormSection
            title="Data & DPO"
            hint="Curated verifiable Hub/JSONL first; Data Designer is opt-in. Smoke uses the CI fixture."
            collapsible
            defaultOpen
          >
            <div className="form-field">
              <label>Preference source</label>
              <select
                value={preferenceSource}
                onChange={(e) => setPreferenceSource(e.target.value)}
              >
                <option value="dataset">Dataset (HF hub / local verifiable)</option>
                <option value="grounded_library">Grounded library (operator JSONL)</option>
                <option value="data_designer">Data Designer (opt-in synth)</option>
                <option value="teacher_style">Teacher≻student (style DPO, not outcome RL)</option>
              </select>
            </div>
            {preferenceSource === "dataset" && (
              <div className="form-field">
                <label>Dataset ref (Hub id or local JSONL)</label>
                <input
                  value={datasetRef}
                  onChange={(e) => setDatasetRef(e.target.value)}
                  placeholder="e.g. open-r1/OpenR1-Math-220k or ~/.seiso/…/prompts.jsonl"
                />
              </div>
            )}
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
                placeholder={
                  preferenceSource === "grounded_library"
                    ? "Required for grounded_library (answer/tests per row)"
                    : "Optional local prompts; Hub id goes in dataset ref"
                }
              />
            </div>
            {configError && <p className="field-error">{configError}</p>}
            {preset === "smoke" && (
              <p className="muted-text studio-field-hint">
                Smoke is a CI fixture only — switch to reproducible/full + a real dataset_ref for
                meaningful Distill-RL.
              </p>
            )}
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
