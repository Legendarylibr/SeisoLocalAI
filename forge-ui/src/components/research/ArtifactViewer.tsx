type ArtifactViewerProps = {
  title?: string;
  data: Record<string, unknown> | null;
  emptyMessage?: string;
};

export function ArtifactViewer({ title, data, emptyMessage = "No artifact data." }: ArtifactViewerProps) {
  if (!data) {
    return <p className="research-empty">{emptyMessage}</p>;
  }

  return (
    <div className="artifact-viewer">
      {title && <div className="artifact-viewer-title">{title}</div>}
      <pre className="artifact-viewer-body">{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}
