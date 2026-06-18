import { useCallback, useState } from "react";
import { subscribeSSE } from "@/lib/api";

type StreamHandlers = {
  onLog?: (line: string) => void;
  onError?: (message: string) => void;
  onResult?: (data: string) => void;
  onEvent?: (event: string, data: string) => void;
};

export function usePipelineJobStream() {
  const [logs, setLogs] = useState<string[]>([]);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [activeJob, setActiveJob] = useState<string | null>(null);

  const resetStream = useCallback(() => {
    setLogs([]);
    setResult(null);
  }, []);

  const watchJob = useCallback((streamPath: string, jobId: string, handlers: StreamHandlers = {}) => {
    setActiveJob(jobId);
    subscribeSSE(streamPath, (event, data) => {
      handlers.onEvent?.(event, data);
      if (event === "log") {
        setLogs((prev) => [...prev, data]);
        handlers.onLog?.(data);
      }
      if (event === "error") {
        const line = `ERROR: ${data}`;
        setLogs((prev) => [...prev, line]);
        handlers.onError?.(data);
      }
      if (event === "result") {
        handlers.onResult?.(data);
        try {
          setResult(JSON.parse(data));
        } catch {
          /* ignore malformed payloads */
        }
      }
    });
  }, []);

  return { logs, result, activeJob, resetStream, watchJob };
}
