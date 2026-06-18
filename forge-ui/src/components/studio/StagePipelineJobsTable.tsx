type StageJobRow = {
  id: string;
  status: string;
  stages?: string[];
  model_dir?: string | null;
  created_at?: string;
};

type StagePipelineJobsTableProps = {
  jobs: StageJobRow[];
  emptyMessage: string;
};

export function StagePipelineJobsTable({ jobs, emptyMessage }: StagePipelineJobsTableProps) {
  return (
    <div className="card" style={{ marginTop: "1rem" }}>
      <h3 className="section-title">Recent jobs</h3>
      {jobs.length === 0 ? (
        <p className="muted-text">{emptyMessage}</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Status</th>
              <th>Stages</th>
              <th>Model</th>
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((j) => (
              <tr key={j.id}>
                <td className="mono">{j.id.slice(0, 8)}…</td>
                <td><span className={`badge badge-${j.status}`}>{j.status}</span></td>
                <td>{j.stages?.join(", ") || "—"}</td>
                <td className="mono">{j.model_dir ? j.model_dir.split("/").slice(-2).join("/") : "—"}</td>
                <td className="muted-cell">{j.created_at?.slice(0, 19)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
