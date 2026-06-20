import { LogStream } from "@/components/research/LogStream";
import { StudioCardHeader } from "@/components/studio/StudioCardHeader";

type PipelineJobPanelProps = {
  activeJob: string | null;
  logs: string[];
  result?: Record<string, unknown> | null;
  emptyMessage?: string;
};

export function PipelineJobPanel({
  activeJob,
  logs,
  result = null,
  emptyMessage = "Logs appear here during the pipeline.",
}: PipelineJobPanelProps) {
  return (
    <div className="card studio-card studio-card-scroll">
      <StudioCardHeader
        icon="②"
        title="Live output"
        description="Streaming logs and pipeline results"
        tone="monitor"
        meta={
          activeJob ? (
            <span className="mono studio-job-id">{activeJob.slice(0, 8)}…</span>
          ) : undefined
        }
      />
      <LogStream logs={logs} emptyMessage={emptyMessage} fill label="Pipeline log" />
      {result && (
        <div className="studio-artifact-section">
          <div className="form-section-head">
            <h3 className="form-section-title">Result</h3>
          </div>
          <pre className="log-panel artifact-viewer-body">{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
