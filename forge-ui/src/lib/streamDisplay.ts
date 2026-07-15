/** Smooth streaming display for multi-pass chat (auto-continue).

 * Server text is the source of truth. The visible draft catches up at a capped
 * rate so large batches after a continue pause do not dump as one jump, while
 * still never lagging far behind a fast stream.
 */

export type StreamPaintState = {
  /** True while waiting for the next multi-pass chunk (context pack / prefill). */
  continuing?: boolean;
};

export type StreamDisplaySink = {
  /** Push the latest full assistant draft (not a delta). */
  push: (text: string) => void;
  /** Mark a multi-pass continue gap (keep text, show continuing cue). */
  noteContinue: () => void;
  /** Force any pending paint/scroll flush to the full server draft. */
  flush: () => void;
  /** Reset for a new generation. */
  reset: () => void;
};

/** Soft cap of characters revealed per animation frame. */
const BASE_CHARS_PER_FRAME = 40;
/** When lag exceeds this, catch up faster so we never trail a whole paragraph. */
const MAX_SOFT_LAG = 480;

export function createStreamDisplaySink(
  onPaint: (text: string, state?: StreamPaintState) => void,
  onScroll?: () => void,
): StreamDisplaySink {
  let serverText = "";
  let displayed = "";
  let painted = "";
  let continuing = false;
  let frame: number | null = null;
  let first = true;

  const paint = (text: string) => {
    if (text === painted && !continuing) return;
    painted = text;
    onPaint(text, { continuing });
    onScroll?.();
  };

  const schedule = () => {
    if (frame !== null) return;
    frame = requestAnimationFrame(tick);
  };

  const tick = () => {
    frame = null;
    const lag = serverText.length - displayed.length;

    if (lag <= 0) {
      displayed = serverText;
      paint(displayed);
      // Continue-gap cue is CSS-animated; no need to re-tick until new text.
      return;
    }

    // Adaptive reveal: slow when nearly caught up, faster when lag is large
    // (typical right after a multi-pass resume dumps a buffered batch).
    let step = BASE_CHARS_PER_FRAME;
    if (lag > MAX_SOFT_LAG) {
      step = lag - Math.floor(MAX_SOFT_LAG / 2);
    } else if (lag > 240) {
      step = 96;
    } else if (lag > 100) {
      step = 64;
    }

    let nextLen = Math.min(serverText.length, displayed.length + step);
    // Prefer ending a frame on a word boundary when close — reduces mid-word flicker.
    if (nextLen < serverText.length && nextLen - displayed.length >= 8) {
      const window = serverText.slice(displayed.length, nextLen + 12);
      const space = window.search(/[\s\n]/);
      if (space >= 0 && space < step + 8) {
        nextLen = displayed.length + space + 1;
      }
    }

    displayed = serverText.slice(0, nextLen);
    // New tokens arrived — leave the continue cue until the next noteContinue.
    if (lag > 0) continuing = false;
    paint(displayed);

    if (displayed.length < serverText.length || continuing) {
      schedule();
    }
  };

  return {
    push(text: string) {
      // Server drafts only grow during a generation; if a full replace arrives
      // (final message event), snap target without shrinking display.
      if (text.length >= serverText.length || text.startsWith(displayed.slice(0, Math.min(32, displayed.length)))) {
        serverText = text;
      } else {
        // Rare out-of-order replace — accept server truth.
        serverText = text;
        if (displayed.length > serverText.length) {
          displayed = serverText;
        }
      }
      // Tokens after a continue gap clear the "waiting" cue.
      if (text.length > displayed.length) {
        continuing = false;
      }
      if (first) {
        first = false;
        // First paint ASAP so TTFT feels snappy.
        displayed = serverText.length <= BASE_CHARS_PER_FRAME ? serverText : serverText.slice(0, BASE_CHARS_PER_FRAME);
        queueMicrotask(() => {
          paint(displayed);
          if (displayed.length < serverText.length) schedule();
        });
        return;
      }
      schedule();
    },
    noteContinue() {
      continuing = true;
      // Keep current text; paint cue without wiping content.
      paint(displayed || serverText);
      schedule();
    },
    flush() {
      if (frame !== null) {
        cancelAnimationFrame(frame);
        frame = null;
      }
      continuing = false;
      displayed = serverText;
      paint(displayed);
    },
    reset() {
      if (frame !== null) {
        cancelAnimationFrame(frame);
        frame = null;
      }
      serverText = "";
      displayed = "";
      painted = "";
      continuing = false;
      first = true;
    },
  };
}
