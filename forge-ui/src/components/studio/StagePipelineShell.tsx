import { type ReactNode } from "react";
import { FormSection } from "@/components/research/FormSection";
import { PipelineJobPanel } from "@/components/studio/PipelineJobPanel";
import { StagePipelineJobsTable } from "@/components/studio/StagePipelineJobsTable";
import { StudioCardBody } from "@/components/studio/StudioCardBody";
import { StudioCardHeader } from "@/components/studio/StudioCardHeader";
import { StudioPageShell } from "@/components/StudioPageShell";

type StageJobRow = {
  id: string;
  status: string;
  stages?: string[];
  model_dir?: string | null;
  error_text?: string | null;
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
  presetsLoading?: boolean;
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
  startError?: string | null;
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
  presetsLoading = false,
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
  startError = null,
  onStart,
  startLabel,
  children,
}: StagePipelineShellProps) {
  return (
    <StudioPageShell title={title} subtitle={subtitle}>
      <div className="train-layout train-layout--config-monitor">
        <div className="card studio-card">
          <StudioCardHeader
            icon={cardIcon}
            title="Pipeline configuration"
            description={cardDesc}
          />

          <StudioCardBody>
          <div className="studio-config-grid">
            <div className="studio-config-block">
              <FormSection title="Pipeline" hint="Preset and active stages.">
                <div className="form-field">
                  <label>Preset</label>
                  <select value={preset} onChange={(e) => setPreset(e.target.value)}>
                    {presetList.map((p) => (
                      <option key={p.id} value={p.id}>{p.label}</option>
                    ))}
                  </select>
                </div>
                {presetsLoading && <p className="field-hint">Loading presets…</p>}
                {!presetsLoading && presetList.length === 0 && (
                  <p className="field-hint">Presets unavailable — check Forge connection.</p>
                )}
                <div className="form-field">
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
                </div>
              </FormSection>
            </div>

            <div className="studio-config-block">{children}</div>
          </div>
          </StudioCardBody>

          {startError && (
            <div className="status-callout status-callout-error studio-error-callout" role="alert">
              <div className="status-callout-body">
                <strong className="status-callout-title">Failed to start pipeline</strong>
                <div className="status-callout-text">{startError}</div>
              </div>
            </div>
          )}

          <div className="studio-action-bar studio-action-bar-flush">
            <button
              className="btn btn-primary btn-lg"
              onClick={onStart}
              disabled={starting || !canStart}
            >
              {starting ? "Starting…" : startLabel}
            </button>
          </div>
        </div>

        <div className="studio-monitor-stack">
          <PipelineJobPanel activeJob={activeJob} logs={logs} result={result} />
          <StagePipelineJobsTable jobs={jobs} emptyMessage={jobsEmptyMessage} />
        </div>
      </div>
    </StudioPageShell>
  );
}
