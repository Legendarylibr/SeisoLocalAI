import { API, formatApiError, getCsrfToken, request } from "./client";

const MAX_SSE_LOG_LINES = 2000;

function parseSSEBlock(block: string): { event: string; data: string } | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const rawLine of block.split("\n")) {
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) {
      let data = line.slice(5);
      if (data.startsWith(" ")) data = data.slice(1);
      dataLines.push(data);
    }
  }
  return dataLines.length ? { event, data: dataLines.join("\n") } : null;
}

async function consumeSSEStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  onBlock: (event: string, data: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        buffer += decoder.decode();
        buffer = buffer.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() || "";
      for (const block of blocks) {
        const parsed = parseSSEBlock(block);
        if (parsed) onBlock(parsed.event, parsed.data);
      }
    }
    const parsed = parseSSEBlock(buffer.trim());
    if (parsed) onBlock(parsed.event, parsed.data);
  } catch (err) {
    if (!signal?.aborted) throw err;
  } finally {
    reader.cancel().catch(() => {});
  }
}

/** Stream SSE from a POST endpoint (cookie session + CSRF). Returns abort handle. */
export function streamPostSSE(
  path: string,
  body: Record<string, unknown>,
  handlers: Record<string, (data: string) => void>,
): { promise: Promise<void>; abort: () => void } {
  const controller = new AbortController();
  const promise = (async () => {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    const csrf = getCsrfToken();
    if (csrf) headers["X-CSRF-Token"] = csrf;

    let res: Response;
    try {
      res = await fetch(`${API}${path}`, {
        method: "POST",
        headers,
        credentials: "include",
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (err) {
      if (controller.signal.aborted) return;
      throw err;
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(formatApiError(err.detail, res.statusText || "Request failed"));
    }

    const reader = res.body?.getReader();
    if (!reader) throw new Error("Streaming response unavailable");

    await consumeSSEStream(
      reader,
      (event, data) => {
        if (handlers[event]) handlers[event](data);
      },
      controller.signal,
    );
  })();

  return { promise, abort: () => controller.abort() };
}

export function subscribeSSE(
  path: string,
  onEvent: (event: string, data: string) => void,
  onError?: (err: Error) => void,
): () => void {
  const controller = new AbortController();

  void (async () => {
    let res: Response;
    try {
      res = await fetch(`${API}${path}`, { credentials: "include", signal: controller.signal });
    } catch (err) {
      if (!controller.signal.aborted) {
        onError?.(err instanceof Error ? err : new Error("SSE connection failed"));
      }
      return;
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      onError?.(new Error(formatApiError(err.detail, res.statusText || "SSE request failed")));
      return;
    }

    const reader = res.body?.getReader();
    if (!reader) {
      onError?.(new Error("SSE stream unavailable"));
      return;
    }

    try {
      await consumeSSEStream(reader, onEvent, controller.signal);
    } catch (err) {
      if (!controller.signal.aborted) {
        onError?.(err instanceof Error ? err : new Error("SSE stream failed"));
      }
    }
  })();

  return () => controller.abort();
}

/** Stream chat completions via SSE (cookie session + CSRF). Returns abort handle. */
export function streamChat(
  body: Record<string, unknown>,
  handlers: {
    onEvent: (event: string, data: string) => void;
    onError?: (message: string) => void;
  },
): { promise: Promise<void>; abort: () => void } {
  const controller = new AbortController();
  const promise = (async () => {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    const csrf = getCsrfToken();
    if (csrf) headers["X-CSRF-Token"] = csrf;

    let res: Response;
    try {
      res = await fetch(`${API}/inference/chat`, {
        method: "POST",
        headers,
        credentials: "include",
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (err) {
      if (controller.signal.aborted) return;
      throw err;
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(formatApiError(err.detail, res.statusText || "Chat request failed"));
    }

    const reader = res.body?.getReader();
    if (!reader) return;

    await consumeSSEStream(reader, handlers.onEvent, controller.signal);
  })();

  return {
    promise,
    abort: () => {
      controller.abort();
      request<{ active_model: string | null }>("/inference/cancel-generation", { method: "POST" }).catch(() => {});
    },
  };
}

/** Append a log line with a bounded buffer (matches server MAX_LOG_LINES). */
export function appendBoundedLog(prev: string[], line: string): string[] {
  const next = [...prev, line];
  return next.length > MAX_SSE_LOG_LINES ? next.slice(-MAX_SSE_LOG_LINES) : next;
}
