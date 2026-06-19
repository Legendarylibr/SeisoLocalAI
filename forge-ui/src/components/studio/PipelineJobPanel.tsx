import { LogStream } from "@/components/research/LogStream";

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
    <div className="card">
      <h3 className="section-title">
        Job log {activeJob ? <span className="badge">{activeJob.slice(0, 8)}</span> : ""}
      </h3>
      <LogStream logs={logs} emptyMessage={emptyMessage} tall />
      {result && (
        <div style={{ marginTop: "1rem" }}>
          <h3 className="section-title">Result</h3>
          <pre className="log-panel" style={{ fontSize: "0.8rem" }}>
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
