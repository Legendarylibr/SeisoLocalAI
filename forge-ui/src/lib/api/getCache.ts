import { request } from "./client";

type CacheEntry = { expires: number; value: unknown };

const store = new Map<string, CacheEntry>();
const inflight = new Map<string, Promise<unknown>>();

/** Drop cached GET responses (exact path or prefix match). */
export function invalidateApiCache(pathPrefix?: string): void {
  if (!pathPrefix) {
    store.clear();
    return;
  }
  for (const key of store.keys()) {
    if (key === pathPrefix || key.startsWith(`${pathPrefix}?`) || key.startsWith(pathPrefix)) {
      store.delete(key);
    }
  }
}

/** GET with in-flight dedup and a short TTL cache for read-heavy endpoints. */
export async function cachedGet<T>(path: string, ttlMs = 30_000): Promise<T> {
  const now = Date.now();
  const hit = store.get(path);
  if (hit && hit.expires > now) {
    return hit.value as T;
  }

  let pending = inflight.get(path);
  if (!pending) {
    pending = request<T>(path)
      .then((value) => {
        store.set(path, { expires: Date.now() + ttlMs, value });
        inflight.delete(path);
        return value;
      })
      .catch((err) => {
        inflight.delete(path);
        throw err;
      });
    inflight.set(path, pending);
  }

  return pending as Promise<T>;
}

/** Refresh a cached GET — bypasses TTL but still dedupes concurrent callers. */
export async function refreshCachedGet<T>(path: string, ttlMs = 30_000): Promise<T> {
  store.delete(path);
  return cachedGet<T>(path, ttlMs);
}
