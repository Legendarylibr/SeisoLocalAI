import { DataTable } from "@/components/research/DataTable";
import { StudioCardHeader } from "@/components/studio/StudioCardHeader";

type StageJobRow = {
  id: string;
  status: string;
  stages?: string[];
  model_dir?: string | null;
  error_text?: string | null;
  created_at?: string;
};

type StagePipelineJobsTableProps = {
  jobs: StageJobRow[];
  emptyMessage: string;
};

export function StagePipelineJobsTable({ jobs, emptyMessage }: StagePipelineJobsTableProps) {
  return (
    <div className="card studio-card studio-card-compact">
      <StudioCardHeader
        icon="③"
        title="Recent jobs"
        description="Past pipeline runs on this machine"
        tone="history"
        meta={
          jobs.length > 0 ? (
            <span className="badge badge-dim">
              {jobs.length} job{jobs.length === 1 ? "" : "s"}
            </span>
          ) : undefined
        }
      />
      <DataTable
        columns={[
          {
            key: "id",
            header: "ID",
            mono: true,
            render: (j) => `${j.id.slice(0, 8)}…`,
          },
          {
            key: "status",
            header: "Status",
            render: (j) => <span className={`badge badge-${j.status}`}>{j.status}</span>,
          },
          {
            key: "error_text",
            header: "Error",
            render: (j) =>
              j.error_text ? (
                <span className="text-danger" title={j.error_text}>
                  {j.error_text.length > 48
                    ? `${j.error_text.slice(0, 48)}…`
                    : j.error_text}
                </span>
              ) : (
                "—"
              ),
          },
          {
            key: "stages",
            header: "Stages",
            render: (j) => j.stages?.join(", ") || "—",
          },
          {
            key: "model",
            header: "Model",
            mono: true,
            render: (j) =>
              j.model_dir ? j.model_dir.split("/").slice(-2).join("/") : "—",
          },
          {
            key: "created_at",
            header: "Created",
            render: (j) => j.created_at?.slice(0, 19) ?? "—",
          },
        ]}
        rows={jobs.slice(0, 6)}
        getRowKey={(j) => j.id}
        emptyMessage={emptyMessage}
      />
    </div>
  );
}
