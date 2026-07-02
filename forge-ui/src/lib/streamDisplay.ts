/** Low-latency streaming display — first chunk immediate, scroll coalesced to rAF. */

export type StreamDisplaySink = {
  /** Push the latest full assistant draft (not a delta). */
  push: (text: string) => void;
  /** Force any pending paint/scroll flush. */
  flush: () => void;
  /** Reset for a new generation. */
  reset: () => void;
};

export function createStreamDisplaySink(
  onPaint: (text: string) => void,
  onScroll?: () => void,
): StreamDisplaySink {
  let latest = "";
  let painted = "";
  let frame: number | null = null;
  let first = true;

  const paint = () => {
    frame = null;
    if (latest === painted) return;
    painted = latest;
    onPaint(latest);
    onScroll?.();
  };

  const schedule = () => {
    if (frame !== null) return;
    frame = requestAnimationFrame(paint);
  };

  return {
    push(text: string) {
      latest = text;
      if (first) {
        first = false;
        queueMicrotask(paint);
        return;
      }
      schedule();
    },
    flush() {
      if (frame !== null) {
        cancelAnimationFrame(frame);
        frame = null;
      }
      paint();
    },
    reset() {
      if (frame !== null) {
        cancelAnimationFrame(frame);
        frame = null;
      }
      latest = "";
      painted = "";
      first = true;
    },
  };
}