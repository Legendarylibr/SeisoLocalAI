type LogStreamProps = {
  title?: string;
  logs: string[];
  emptyMessage?: string;
  tall?: boolean;
};

export function LogStream({ title, logs, emptyMessage = "Waiting for output…", tall = false }: LogStreamProps) {
  return (
    <div className="log-stream">
      {title && <h3 className="log-stream-title">{title}</h3>}
      <div className={`log-panel${tall ? " log-panel-tall" : ""}`}>
        {logs.length > 0 ? logs.join("\n") : emptyMessage}
      </div>
    </div>
  );
}
