import { useEffect, useRef } from "react";

type LogStreamProps = {
  title?: string;
  logs: string[];
  emptyMessage?: string;
  tall?: boolean;
};

/** Strip common ANSI escape sequences for cleaner terminal display. */
function stripAnsi(text: string): string {
  return text.replace(/\x1b\[[0-9;]*[A-Za-z]/g, "");
}

export function LogStream({ title, logs, emptyMessage = "Waiting for output…", tall = false }: LogStreamProps) {
  const panelRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    const el = panelRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs]);

  const text = logs.length > 0 ? logs.map(stripAnsi).join("\n") : emptyMessage;

  return (
    <div className="log-stream">
      {title && <h3 className="log-stream-title">{title}</h3>}
      <pre ref={panelRef} className={`log-panel${tall ? " log-panel-tall" : ""}`}>
        {text}
      </pre>
    </div>
  );
}
