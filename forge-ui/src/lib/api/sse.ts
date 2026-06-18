import { API, formatApiError, getCsrfToken, request } from "./client";

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
    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() || "";
        for (const block of blocks) {
          let event = "message";
          let data = "";
          for (const line of block.split("\n")) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            if (line.startsWith("data:")) data = line.slice(5).trim();
          }
          if (data && handlers[event]) handlers[event](data);
        }
      }
    } catch (err) {
      if (!controller.signal.aborted) throw err;
    } finally {
      reader.cancel().catch(() => {});
    }
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

    const decoder = new TextDecoder();
    let buffer = "";
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        for (const block of parts) {
          let event = "message";
          let data = "";
          for (const line of block.split("\n")) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            if (line.startsWith("data:")) data = line.slice(5).trim();
          }
          if (data) onEvent(event, data);
        }
      }
    } catch (err) {
      if (!controller.signal.aborted) {
        onError?.(err instanceof Error ? err : new Error("SSE stream failed"));
      }
    } finally {
      reader.cancel().catch(() => {});
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
      throw new Error(err.detail || "Chat request failed");
    }

    const reader = res.body?.getReader();
    if (!reader) return;
    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() || "";
        for (const block of blocks) {
          let event = "message";
          let data = "";
          for (const line of block.split("\n")) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            if (line.startsWith("data:")) data = line.slice(5).trim();
          }
          if (data) handlers.onEvent(event, data);
        }
      }
    } catch (err) {
      if (!controller.signal.aborted) throw err;
    } finally {
      reader.cancel().catch(() => {});
    }
  })();

  return {
    promise,
    abort: () => {
      controller.abort();
      request<{ active_model: string | null }>("/inference/cancel", { method: "POST" }).catch(() => {});
    },
  };
}
