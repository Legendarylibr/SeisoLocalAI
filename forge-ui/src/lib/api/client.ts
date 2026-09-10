const TAURI_API_BASE = "http://127.0.0.1:8765/api";

// In the Tauri desktop webview (tauri://localhost) the relative /api path
// resolves to the custom protocol, not the Forge backend. Use the absolute
// backend URL there; the web build stays same-origin.
export const API =
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window
    ? TAURI_API_BASE
    : "/api";

const MUTATING = new Set(["POST", "PUT", "DELETE", "PATCH"]);

/**
 * Hard cap on a single non-streaming API call. Guards the chat UI against a
 * wedged backend (slow HF sync, hung model unload) leaving it stuck in a
 * loading state forever. Streaming endpoints use their own connection timeout
 * and are not bounded here.
 */
const DEFAULT_REQUEST_TIMEOUT_MS = 60_000;

/** Read CSRF double-submit cookie set by the server on login/register. */
export function getCsrfToken(): string | null {
  const match = document.cookie.match(/(?:^|;\s*)seiso_csrf=([^;]*)/);
  if (!match?.[1]) return null;
  try {
    return decodeURIComponent(match[1]);
  } catch {
    return null;
  }
}

/** Clear any legacy localStorage tokens from older builds. */
export function clearLegacyToken() {
  try {
    localStorage.removeItem("seiso_token");
  } catch {
    /* ignore */
  }
}

function formatApiError(detail: unknown, fallback = "Request failed"): string {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object" && "msg" in item) {
          return String((item as { msg?: unknown }).msg ?? "");
        }
        return "";
      })
      .filter(Boolean);
    if (parts.length) return parts.join("; ");
  }
  if (detail && typeof detail === "object" && "msg" in detail) {
    return String((detail as { msg?: unknown }).msg ?? fallback);
  }
  return fallback;
}

/** RequestInit extended with an optional per-call timeout override. */
export type ApiRequestInit = RequestInit & { timeoutMs?: number };

export async function request<T>(path: string, init: ApiRequestInit = {}): Promise<T> {
  const method = (init.method || "GET").toUpperCase();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(init.headers as Record<string, string>),
  };
  if (MUTATING.has(method)) {
    const csrf = getCsrfToken();
    if (csrf) headers["X-CSRF-Token"] = csrf;
  }

  const controller = new AbortController();
  const timeoutMs =
    typeof init.timeoutMs === "number" ? init.timeoutMs : DEFAULT_REQUEST_TIMEOUT_MS;
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API}${path}`, {
      ...init,
      headers,
      credentials: "include",
      signal: controller.signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      const detail = formatApiError(err.detail, res.statusText || "Request failed");
      if (res.status === 403 && /csrf/i.test(detail)) {
        throw new Error("Session security token expired — sign out and sign in again, then retry.");
      }
      throw new Error(detail);
    }
    return res.json() as Promise<T>;
  } finally {
    window.clearTimeout(timer);
  }
}

export { formatApiError };
