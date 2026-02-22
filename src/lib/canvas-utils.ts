/**
 * Shared canvas utilities used by canvas-chart, canvas-heatmap, and simulation components.
 */

/** Extract plain text from a string-or-object title (Plotly-compatible). */
export function extractText(v?: string | { text: string; font?: unknown }): string {
  if (!v) return '';
  if (typeof v === 'string') return v;
  return v.text || '';
}

/** Format a numeric axis label. */
export function fmtNum(v: number): string {
  if (v === 0) return '0';
  const abs = Math.abs(v);
  if (abs >= 1e6 || (abs < 0.01 && abs > 0)) return v.toExponential(1);
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 1) return v.toFixed(1);
  return v.toPrecision(3);
}

/**
 * Set up a canvas for high-DPI rendering.
 * Returns the 2D context scaled to the device pixel ratio, or null if unavailable.
 */
export function setupCanvasDpi(
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
): CanvasRenderingContext2D | null {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.scale(dpr, dpr);
  return ctx;
}

/**
 * Create a ResizeObserver that calls `redraw` whenever `container` resizes.
 * Returns a cleanup function.
 */
export function createCanvasResizeObserver(
  container: HTMLElement,
  redraw: () => void,
): () => void {
  const ro = new ResizeObserver(redraw);
  ro.observe(container);
  return () => ro.disconnect();
}
