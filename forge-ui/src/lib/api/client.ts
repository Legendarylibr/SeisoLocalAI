export const API = "/api";

const MUTATING = new Set(["POST", "PUT", "DELETE", "PATCH"]);

/** Read CSRF double-submit cookie set by the server on login/register. */
export function getCsrfToken(): string | null {
  const match = document.cookie.match(/(?:^|;\s*)seiso_csrf=([^;]*)/);
  return match ? decodeURIComponent(match[1]) : null;
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

export async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const method = (init.method || "GET").toUpperCase();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(init.headers as Record<string, string>),
  };
  if (MUTATING.has(method)) {
    const csrf = getCsrfToken();
    if (csrf) headers["X-CSRF-Token"] = csrf;
  }

  const res = await fetch(`${API}${path}`, { ...init, headers, credentials: "include" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = formatApiError(err.detail, res.statusText || "Request failed");
    if (res.status === 403 && /csrf/i.test(detail)) {
      throw new Error("Session security token expired — sign out and sign in again, then retry.");
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export { formatApiError };
