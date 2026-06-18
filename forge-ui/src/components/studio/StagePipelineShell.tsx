import { type ReactNode } from "react";
import { PipelineJobPanel } from "@/components/studio/PipelineJobPanel";
import { StagePipelineJobsTable } from "@/components/studio/StagePipelineJobsTable";
import { StudioPageShell } from "@/components/StudioPageShell";

type StageJobRow = {
  id: string;
  status: string;
  stages?: string[];
  model_dir?: string | null;
  created_at?: string;
};

type StagePipelineShellProps = {
  title: string;
  subtitle: string;
  cardIcon: string;
  cardDesc: string;
  preset: string;
  setPreset: (id: string) => void;
  presetList: Array<{ id: string; label: string }>;
  allStages: string[];
  stageHelp: Record<string, string>;
  selectedStages: string[];
  toggleStage: (stage: string) => void;
  logs: string[];
  result: Record<string, unknown> | null;
  activeJob: string | null;
  jobs: StageJobRow[];
  jobsEmptyMessage: string;
  canStart: boolean;
  starting: boolean;
  onStart: () => void;
  startLabel: string;
  children: ReactNode;
};

export function StagePipelineShell({
  title,
  subtitle,
  cardIcon,
  cardDesc,
  preset,
  setPreset,
  presetList,
  allStages,
  stageHelp,
  selectedStages,
  toggleStage,
  logs,
  result,
  activeJob,
  jobs,
  jobsEmptyMessage,
  canStart,
  starting,
  onStart,
  startLabel,
  children,
}: StagePipelineShellProps) {
  return (
    <StudioPageShell title={title} subtitle={subtitle}>
      <div className="train-layout">
        <div className="card compress-config-card studio-card">
          <div className="studio-card-head">
            <span className="studio-card-icon" aria-hidden>{cardIcon}</span>
            <div className="studio-card-head-text">
              <div className="studio-card-title">Pipeline configuration</div>
              <div className="studio-card-desc">{cardDesc}</div>
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

          {children}

          <button
            className="btn btn-primary btn-lg studio-action-bar-standalone"
            onClick={onStart}
            disabled={starting || !canStart}
          >
            {starting ? "Starting…" : startLabel}
          </button>
        </div>

        <PipelineJobPanel activeJob={activeJob} logs={logs} result={result} />
      </div>

      <StagePipelineJobsTable jobs={jobs} emptyMessage={jobsEmptyMessage} />
    </StudioPageShell>
  );
}
