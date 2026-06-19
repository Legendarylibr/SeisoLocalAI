/** Throw AbortError when a signal has been cancelled. */
export function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new DOMException("Aborted", "AbortError");
  }
}

/** Abort an operation when an external signal fires. */
export function bindAbort<T>(
  signal: AbortSignal | undefined,
  abort: () => void,
  promise: Promise<T>,
): Promise<T> {
  if (!signal) return promise;
  if (signal.aborted) {
    abort();
    return Promise.reject(new DOMException("Aborted", "AbortError"));
  }
  const onAbort = () => abort();
  signal.addEventListener("abort", onAbort, { once: true });
  return promise.finally(() => signal.removeEventListener("abort", onAbort));
}
