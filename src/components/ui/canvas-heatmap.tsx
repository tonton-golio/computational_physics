'use client';

import React, { useRef, useEffect, useState } from 'react';
import { type CanvasTheme, DARK_THEME, getCanvasTheme } from '@/lib/canvas-theme';

// ── Colorscales ────────────────────────────────────────────────────────

type ColorStop = [number, [number, number, number]];

const COLORSCALES: Record<string, ColorStop[]> = {
  Viridis: [
    [0, [68, 1, 84]], [0.25, [59, 82, 139]], [0.5, [33, 145, 140]],
    [0.75, [94, 201, 98]], [1, [253, 231, 37]],
  ],
  RdBu: [
    [0, [103, 0, 31]], [0.25, [214, 96, 77]], [0.5, [247, 247, 247]],
    [0.75, [67, 147, 195]], [1, [5, 48, 97]],
  ],
  Hot: [
    [0, [10, 0, 0]], [0.33, [200, 20, 0]], [0.66, [255, 200, 0]], [1, [255, 255, 255]],
  ],
  Inferno: [
    [0, [0, 0, 4]], [0.25, [87, 16, 110]], [0.5, [188, 55, 84]],
    [0.75, [249, 142, 9]], [1, [252, 255, 164]],
  ],
  Greys: [
    [0, [0, 0, 0]], [1, [255, 255, 255]],
  ],
  Portland: [
    [0, [12, 51, 131]], [0.25, [10, 136, 186]], [0.5, [242, 211, 56]],
    [0.75, [242, 143, 56]], [1, [217, 30, 30]],
  ],
  Plasma: [
    [0, [13, 8, 135]], [0.25, [126, 3, 168]], [0.5, [204, 71, 120]],
    [0.75, [248, 149, 64]], [1, [240, 249, 33]],
  ],
};

function interpolateColor(stops: ColorStop[], t: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  let lo = stops[0];
  let hi = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (clamped >= stops[i][0] && clamped <= stops[i + 1][0]) {
      lo = stops[i];
      hi = stops[i + 1];
      break;
    }
  }
  const range = hi[0] - lo[0] || 1;
  const f = (clamped - lo[0]) / range;
  return [
    Math.round(lo[1][0] + (hi[1][0] - lo[1][0]) * f),
    Math.round(lo[1][1] + (hi[1][1] - lo[1][1]) * f),
    Math.round(lo[1][2] + (hi[1][2] - lo[1][2]) * f),
  ];
}

function parseColorscale(
  cs?: string | Array<[number, string]>,
): ColorStop[] {
  if (!cs) return COLORSCALES.Viridis;
  if (typeof cs === 'string') return COLORSCALES[cs] || COLORSCALES.Viridis;
  // Custom array format: [[0, 'rgb(r,g,b)'], [1, 'rgb(r,g,b)']]
  return cs.map(([t, color]) => {
    const m = color.match(/(\d+)/g);
    if (m && m.length >= 3) return [t, [+m[0], +m[1], +m[2]]] as ColorStop;
    return [t, [128, 128, 128]] as ColorStop;
  });
}

// ── Theme ──────────────────────────────────────────────────────────────

let THEME: CanvasTheme = DARK_THEME;

// ── Types ──────────────────────────────────────────────────────────────

interface AxisTitle {
  text: string;
  font?: { size?: number; color?: string };
}

export interface HeatmapData {
  z: number[][];
  x?: (number | string)[];
  y?: (number | string)[];
  type?: 'heatmap' | 'contour' | 'histogram2d';
  colorscale?: string | Array<[number, string]>;
  showscale?: boolean;
  zmin?: number;
  zmax?: number;
  reversescale?: boolean;
  colorbar?: Record<string, unknown>;
  name?: string;
}

interface AnnotationConfig {
  x: number | string;
  y: number | string;
  text: string;
  showarrow?: boolean;
  font?: { size?: number; color?: string };
  xref?: string;
  yref?: string;
}

export interface HeatmapLayout {
  title?: string | { text: string; font?: { size?: number } };
  xaxis?: { title?: string | AxisTitle; [key: string]: unknown };
  yaxis?: { title?: string | AxisTitle; [key: string]: unknown };
  margin?: { t?: number; r?: number; b?: number; l?: number };
  annotations?: AnnotationConfig[];
  [key: string]: unknown;
}

export interface CanvasHeatmapProps {
  data: (HeatmapData | { [key: string]: unknown })[];
  layout?: HeatmapLayout;
  style?: React.CSSProperties;
}

// ── Utility ────────────────────────────────────────────────────────────

function extractText(v?: string | { text: string; font?: unknown }): string {
  if (!v) return '';
  if (typeof v === 'string') return v;
  return v.text || '';
}

function fmtNum(v: number): string {
  if (v === 0) return '0';
  const abs = Math.abs(v);
  if (abs >= 1000) return v.toFixed(0);
  if (abs >= 1) return v.toFixed(1);
  return v.toPrecision(2);
}

// ── Draw ───────────────────────────────────────────────────────────────

function draw(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  data: CanvasHeatmapProps['data'],
  layout?: HeatmapLayout,
) {
  const mg = {
    t: layout?.margin?.t ?? 40,
    r: layout?.margin?.r ?? 50,
    b: layout?.margin?.b ?? 50,
    l: layout?.margin?.l ?? 60,
  };
  const showScale = (data[0] as HeatmapData)?.showscale !== false;
  const colorbarWidth = showScale ? 40 : 0;
  const pw = w - mg.l - mg.r - colorbarWidth;
  const ph = h - mg.t - mg.b;
  if (pw <= 0 || ph <= 0) return;

  // Clear
  ctx.fillStyle = THEME.bg;
  ctx.fillRect(0, 0, w, h);

  // Get first heatmap data
  const hm = data[0] as HeatmapData;
  if (!hm?.z || hm.z.length === 0) return;

  const rows = hm.z.length;
  const cols = hm.z[0]?.length ?? 0;
  if (cols === 0) return;

  // Value range
  let zMin = hm.zmin ?? Infinity;
  let zMax = hm.zmax ?? -Infinity;
  if (!isFinite(zMin) || !isFinite(zMax)) {
    for (const row of hm.z) {
      for (const v of row) {
        if (isFinite(v)) {
          if (v < zMin) zMin = v;
          if (v > zMax) zMax = v;
        }
      }
    }
  }
  const zRange = zMax - zMin || 1;

  const stops = parseColorscale(hm.colorscale);

  // Draw cells
  const cellW = pw / cols;
  const cellH = ph / rows;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = hm.z[r][c];
      let t = (v - zMin) / zRange;
      if (hm.reversescale) t = 1 - t;
      const [cr, cg, cb] = interpolateColor(stops, t);
      ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
      ctx.fillRect(mg.l + c * cellW, mg.t + r * cellH, Math.ceil(cellW) + 1, Math.ceil(cellH) + 1);
    }
  }

  // Annotations (cell values)
  if (layout?.annotations) {
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = THEME.fontAnnotation;
    for (const ann of layout.annotations) {
      const col = typeof ann.x === 'number' ? ann.x : 0;
      const row = typeof ann.y === 'number' ? ann.y : 0;
      const px = mg.l + (col + 0.5) * cellW;
      const py = mg.t + (row + 0.5) * cellH;
      ctx.fillStyle = ann.font?.color || THEME.textStrong;
      ctx.fillText(String(ann.text), px, py);
    }
  }

  // X labels
  if (hm.x) {
    ctx.fillStyle = THEME.text;
    ctx.font = THEME.fontSmall;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const step = Math.max(1, Math.floor(cols / 10));
    for (let c = 0; c < cols; c += step) {
      ctx.fillText(String(hm.x[c] ?? c), mg.l + (c + 0.5) * cellW, mg.t + ph + 4);
    }
  }

  // Y labels
  if (hm.y) {
    ctx.fillStyle = THEME.text;
    ctx.font = THEME.fontSmall;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const step = Math.max(1, Math.floor(rows / 10));
    for (let r = 0; r < rows; r += step) {
      ctx.fillText(String(hm.y[r] ?? r), mg.l - 6, mg.t + (r + 0.5) * cellH);
    }
  }

  // Colorbar
  if (showScale) {
    const cbX = mg.l + pw + 10;
    const cbW = 16;
    const cbH = ph;
    for (let py = 0; py < cbH; py++) {
      const t = 1 - py / cbH;
      const [cr, cg, cb] = interpolateColor(stops, hm.reversescale ? 1 - t : t);
      ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
      ctx.fillRect(cbX, mg.t + py, cbW, 2);
    }
    ctx.strokeStyle = THEME.axis;
    ctx.lineWidth = 1;
    ctx.strokeRect(cbX, mg.t, cbW, cbH);
    // Colorbar labels
    ctx.fillStyle = THEME.text;
    ctx.font = THEME.fontSmall;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(fmtNum(zMax), cbX + cbW + 3, mg.t);
    ctx.textBaseline = 'bottom';
    ctx.fillText(fmtNum(zMin), cbX + cbW + 3, mg.t + cbH);
    ctx.textBaseline = 'middle';
    ctx.fillText(fmtNum((zMin + zMax) / 2), cbX + cbW + 3, mg.t + cbH / 2);
  }

  // Border
  ctx.strokeStyle = THEME.axis;
  ctx.lineWidth = 1;
  ctx.strokeRect(mg.l, mg.t, pw, ph);

  // Axis titles
  const xTitle = extractText(layout?.xaxis?.title as string | { text: string });
  const yTitle = extractText(layout?.yaxis?.title as string | { text: string });
  ctx.fillStyle = THEME.text;
  ctx.font = THEME.font;
  if (xTitle) {
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(xTitle, mg.l + pw / 2, mg.t + ph + 28);
  }
  if (yTitle) {
    ctx.save();
    ctx.translate(14, mg.t + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(yTitle, 0, 0);
    ctx.restore();
  }

  // Title
  const title = extractText(layout?.title);
  if (title) {
    ctx.fillStyle = THEME.textStrong;
    ctx.font = THEME.fontTitle;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(title, mg.l + pw / 2, mg.t - 8);
  }
}

// ── React Component ────────────────────────────────────────────────────

export function CanvasHeatmap({ data, layout, style }: CanvasHeatmapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [themeKey, setThemeKey] = useState(0);

  useEffect(() => {
    const observer = new MutationObserver(() => setThemeKey(k => k + 1));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const redraw = () => {
      THEME = getCanvasTheme();
      const width = container.clientWidth;
      const height = parseInt(String(style?.height || 320), 10);
      const dpr = window.devicePixelRatio || 1;

      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.scale(dpr, dpr);
      draw(ctx, width, height, data, layout);
    };

    redraw();

    const ro = new ResizeObserver(redraw);
    ro.observe(container);
    return () => ro.disconnect();
  }, [data, layout, style, themeKey]);

  return (
    <div ref={containerRef} style={{ width: style?.width || '100%' }}>
      <canvas ref={canvasRef} style={{ display: 'block', borderRadius: '4px' }} />
    </div>
  );
}
