import { useCallback, useEffect, useRef, useState } from "react";
import { appendBoundedLog, subscribeSSE } from "@/lib/api/sse";

type StreamHandlers = {
  onLog?: (line: string) => void;
  onError?: (message: string) => void;
  onResult?: (data: string) => void;
  onEvent?: (event: string, data: string) => void;
  onStreamError?: (message: string) => void;
};

export function usePipelineJobStream() {
  const [logs, setLogs] = useState<string[]>([]);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [activeJob, setActiveJob] = useState<string | null>(null);
  const abortRef = useRef<(() => void) | null>(null);

  const stopStream = useCallback(() => {
    abortRef.current?.();
    abortRef.current = null;
  }, []);

  const resetStream = useCallback(() => {
    stopStream();
    setLogs([]);
    setResult(null);
  }, [stopStream]);

  const watchJob = useCallback((streamPath: string, jobId: string, handlers: StreamHandlers = {}) => {
    stopStream();
    setActiveJob(jobId);
    abortRef.current = subscribeSSE(
      streamPath,
      (event, data) => {
        handlers.onEvent?.(event, data);
        if (event === "log") {
          setLogs((prev) => appendBoundedLog(prev, data));
          handlers.onLog?.(data);
        }
        if (event === "error") {
          const line = `ERROR: ${data}`;
          setLogs((prev) => appendBoundedLog(prev, line));
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
      },
      (err) => {
        const line = `ERROR: ${err.message}`;
        setLogs((prev) => appendBoundedLog(prev, line));
        handlers.onStreamError?.(err.message);
      },
    );
  }, [stopStream]);

  useEffect(() => () => stopStream(), [stopStream]);

  return { logs, result, activeJob, resetStream, watchJob };
}
