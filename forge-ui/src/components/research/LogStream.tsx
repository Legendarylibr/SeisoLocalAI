import { useEffect, useRef } from "react";

type LogStreamProps = {
  title?: string;
  label?: string;
  logs: string[];
  emptyMessage?: string;
  tall?: boolean;
  fill?: boolean;
};

/** Strip common ANSI escape sequences for cleaner terminal display. */
function stripAnsi(text: string): string {
  return text.replace(/\x1b\[[0-9;]*[A-Za-z]/g, "");
}

export function LogStream({
  title,
  label,
  logs,
  emptyMessage = "Waiting for output…",
  tall = false,
  fill = false,
}: LogStreamProps) {
  const panelRef = useRef<HTMLPreElement>(null);
  const showChrome = fill || tall;
  const chromeLabel = label || title || "Output";

  useEffect(() => {
    const el = panelRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs]);

  const text = logs.length > 0 ? logs.map(stripAnsi).join("\n") : emptyMessage;
  const panelClass = [
    "log-panel",
    tall ? "log-panel-tall" : "",
    fill ? "log-panel-fill" : "",
    showChrome ? "log-panel-chrome" : "",
  ].filter(Boolean).join(" ");

  return (
    <div className={`log-stream${fill ? " log-stream-fill" : ""}`}>
      {title && !showChrome && <h3 className="log-stream-title">{title}</h3>}
      {showChrome && (
        <div className="log-stream-chrome">
          <span className="log-stream-dot" aria-hidden />
          <span className="log-stream-chrome-label">{chromeLabel}</span>
          {logs.length > 0 && (
            <span className="log-stream-chrome-meta">{logs.length} lines</span>
          )}
        </div>
      )}
      <pre ref={panelRef} className={panelClass}>
        {text}
      </pre>
    </div>
  );
}
