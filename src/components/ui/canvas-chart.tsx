"use client";

import React, { useRef, useEffect } from 'react';
import { type CanvasTheme, DARK_THEME, getCanvasTheme } from '@/lib/canvas-theme';
import { extractText, fmtNum } from '@/lib/canvas-utils';
import { useSimulationFullscreen } from '@/lib/simulation-fullscreen-context';
import { useIsInsideMain } from '@/components/ui/simulation-main';
import { useThemeChangeKey } from '@/lib/use-theme';

// ── Types (Plotly-compatible for easy migration) ───────────────────────

interface TraceLine {
  color?: string;
  width?: number;
  dash?: 'solid' | 'dash' | 'dot' | 'dashdot' | string;
}

interface TraceMarker {
  color?: string | string[] | number[];
  size?: number;
  symbol?: string;
  opacity?: number;
  colorscale?: string;
  line?: { width?: number; color?: string };
}

export interface ChartTrace {
  x?: number[] | string[];
  y?: number[];
  type?: 'scatter' | 'histogram' | 'bar' | (string & Record<never, never>);
  mode?: 'lines' | 'markers' | 'lines+markers' | 'text' | 'markers+text' | (string & Record<never, never>);
  line?: TraceLine;
  marker?: TraceMarker;
  fill?: 'tozeroy' | 'tonexty' | 'toself' | (string & Record<never, never>);
  fillcolor?: string;
  name?: string;
  nbinsx?: number;
  text?: string[];
  textposition?: string;
  orientation?: 'v' | 'h';
  opacity?: number;
  hoverinfo?: string;
  showlegend?: boolean;
  width?: number | number[];
  [key: string]: unknown;
}

interface AxisTitle {
  text: string;
  font?: { size?: number; color?: string };
}

interface AxisConfig {
  title?: string | AxisTitle;
  type?: 'linear' | 'log' | '-';
  range?: number[];
  tickvals?: number[];
  ticktext?: (string | number)[];
  showgrid?: boolean;
  zeroline?: boolean;
  dtick?: number;
  [key: string]: unknown;
}

interface ShapeConfig {
  type?: 'line' | 'rect';
  x0?: number;
  x1?: number;
  y0?: number | string;
  y1?: number | string;
  xref?: string;
  yref?: string;
  line?: { color?: string; width?: number; dash?: string };
}

export interface ChartLayout {
  title?: string | { text: string; font?: { size?: number; color?: string } };
  xaxis?: AxisConfig;
  yaxis?: AxisConfig;
  xaxis2?: AxisConfig;
  yaxis2?: AxisConfig;
  showlegend?: boolean;
  shapes?: ShapeConfig[];
  barmode?: 'group' | 'stack' | 'overlay';
  bargap?: number;
  margin?: { t?: number; r?: number; b?: number; l?: number };
  annotations?: Record<string, unknown>[];
  [key: string]: unknown;
}

export interface CanvasChartProps {
  data: ChartTrace[];
  layout?: ChartLayout;
  style?: React.CSSProperties;
}

// ── Theme ──────────────────────────────────────────────────────────────

// Keep a mutable reference so draw() always uses the current theme
let THEME: CanvasTheme = DARK_THEME;

const PALETTE = [
  '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
  '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
];

// ── Utility ────────────────────────────────────────────────────────────

function niceNum(range: number, round: boolean): number {
  const exp = Math.floor(Math.log10(range));
  const frac = range / Math.pow(10, exp);
  let nice: number;
  if (round) {
    nice = frac < 1.5 ? 1 : frac < 3 ? 2 : frac < 7 ? 5 : 10;
  } else {
    nice = frac <= 1 ? 1 : frac <= 2 ? 2 : frac <= 5 ? 5 : 10;
  }
  return nice * Math.pow(10, exp);
}

function genTicks(min: number, max: number, maxTicks = 7): number[] {
  if (min === max) return [min];
  const range = niceNum(max - min, false);
  const step = niceNum(range / (maxTicks - 1), true);
  const start = Math.floor(min / step) * step;
  const ticks: number[] = [];
  for (let t = start; t <= max + step * 0.01; t += step) {
    if (t >= min - step * 0.01 && t <= max + step * 0.01) {
      ticks.push(parseFloat(t.toPrecision(12)));
    }
  }
  return ticks;
}

function genLogTicks(min: number, max: number): number[] {
  const logMin = Math.floor(Math.log10(min));
  const logMax = Math.ceil(Math.log10(max));
  const ticks: number[] = [];
  for (let e = logMin; e <= logMax; e++) ticks.push(Math.pow(10, e));
  return ticks;
}

function fmtLogNum(v: number): string {
  const exp = Math.round(Math.log10(v));
  if (exp === 0) return '1';
  if (exp === 1) return '10';
  return `10^${exp}`;
}

function setDash(ctx: CanvasRenderingContext2D, dash?: string) {
  switch (dash) {
    case 'dash':
      ctx.setLineDash([8, 4]);
      break;
    case 'dot':
      ctx.setLineDash([2, 4]);
      break;
    case 'dashdot':
      ctx.setLineDash([8, 4, 2, 4]);
      break;
    default:
      ctx.setLineDash([]);
  }
}

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ── Histogram binning ──────────────────────────────────────────────────

function histBins(values: number[], nbins: number): { edges: number[]; counts: number[] } {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const step = range / nbins;
  const edges: number[] = [];
  const counts = new Array(nbins).fill(0);
  for (let i = 0; i <= nbins; i++) edges.push(min + i * step);
  for (const v of values) {
    let bin = Math.floor((v - min) / step);
    if (bin >= nbins) bin = nbins - 1;
    if (bin < 0) bin = 0;
    counts[bin]++;
  }
  return { edges, counts };
}

// ── Main draw ──────────────────────────────────────────────────────────

function draw(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  data: ChartTrace[],
  layout?: ChartLayout,
) {
  const mg = {
    t: layout?.margin?.t ?? 40,
    r: layout?.margin?.r ?? 20,
    b: layout?.margin?.b ?? 50,
    l: layout?.margin?.l ?? 60,
  };
  const pw = w - mg.l - mg.r;
  const ph = h - mg.t - mg.b;
  if (pw <= 0 || ph <= 0) return;

  const xLog = layout?.xaxis?.type === 'log';
  const yLog = layout?.yaxis?.type === 'log';

  // ── Collect all data points to find range ────────────────────────
  let allX: number[] = [];
  let allY: number[] = [];
  const processedTraces: Array<{
    trace: ChartTrace;
    xs: number[];
    ys: number[];
    isHist: boolean;
    histData?: { edges: number[]; counts: number[] };
    isBar: boolean;
  }> = [];

  for (const trace of data) {
    const isHist = trace.type === 'histogram';
    const isBar = trace.type === 'bar';

    if (isHist) {
      const vals = (trace.x as number[]) || trace.y;
      const nbins = trace.nbinsx || 20;
      const hd = histBins(vals, nbins);
      const xs = hd.edges.slice(0, -1).map((e, i) => (e + hd.edges[i + 1]) / 2);
      // Normalize to probability density if requested
      let ys = hd.counts;
      if ((trace as Record<string, unknown>).histnorm === 'probability density') {
        const n = vals.length;
        ys = hd.counts.map((c, i) => {
          const binW = hd.edges[i + 1] - hd.edges[i];
          return binW > 0 ? c / (n * binW) : 0;
        });
        hd.counts = ys;
      }
      allX.push(...hd.edges);
      allY.push(0, ...ys);
      processedTraces.push({ trace, xs, ys, isHist: true, histData: hd, isBar: false });
    } else if (isBar) {
      const ys = trace.y || [];
      const xs = (trace.x as number[]) || ys.map((_, i) => i);
      allX.push(...xs);
      allY.push(0, ...ys);
      processedTraces.push({ trace, xs, ys, isHist: false, isBar: true });
    } else {
      const ys = trace.y || [];
      const xs = (trace.x as number[]) || ys.map((_, i) => i);
      allX.push(...xs);
      allY.push(...ys);
      if (trace.fill === 'tozeroy') allY.push(0);
      processedTraces.push({ trace, xs, ys, isHist: false, isBar: false });
    }
  }

  // Filter out non-finite for range
  allX = allX.filter(v => isFinite(v) && (!xLog || v > 0));
  allY = allY.filter(v => isFinite(v) && (!yLog || v > 0));

  if (allX.length === 0 || allY.length === 0) return;

  let xMin = layout?.xaxis?.range?.[0] ?? Math.min(...allX);
  let xMax = layout?.xaxis?.range?.[1] ?? Math.max(...allX);
  let yMin = layout?.yaxis?.range?.[0] ?? Math.min(...allY);
  let yMax = layout?.yaxis?.range?.[1] ?? Math.max(...allY);

  // Add padding (log axes use log-scale padding to avoid pushing min to ~0)
  if (!layout?.xaxis?.range) {
    if (xLog) {
      const logMin = Math.log10(xMin);
      const logMax = Math.log10(xMax);
      const logPad = (logMax - logMin) * 0.05 || 0.5;
      xMin = Math.pow(10, logMin - logPad);
      xMax = Math.pow(10, logMax + logPad);
    } else {
      const xPad = (xMax - xMin) * 0.04 || 1;
      xMin -= xPad;
      xMax += xPad;
    }
  }
  if (!layout?.yaxis?.range) {
    if (yLog) {
      const logMin = Math.log10(yMin);
      const logMax = Math.log10(yMax);
      const logPad = (logMax - logMin) * 0.05 || 0.5;
      yMin = Math.pow(10, logMin - logPad);
      yMax = Math.pow(10, logMax + logPad);
    } else {
      const yPad = (yMax - yMin) * 0.06 || 1;
      yMin -= yPad;
      yMax += yPad;
    }
  }

  if (xLog) { xMax = Math.max(xMax, xMin * 10); }
  if (yLog) { yMax = Math.max(yMax, yMin * 10); }

  // ── Mapping functions ────────────────────────────────────────────
  const mapX = (v: number) => {
    if (xLog) return mg.l + ((Math.log10(Math.max(v, xMin)) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin))) * pw;
    return mg.l + ((v - xMin) / (xMax - xMin)) * pw;
  };
  const mapY = (v: number) => {
    if (yLog) return mg.t + ph - ((Math.log10(Math.max(v, yMin)) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * ph;
    return mg.t + ph - ((v - yMin) / (yMax - yMin)) * ph;
  };

  // ── Background ───────────────────────────────────────────────────
  ctx.fillStyle = THEME.bg;
  ctx.fillRect(mg.l, mg.t, pw, ph);

  // ── Grid ─────────────────────────────────────────────────────────
  const xTicks = xLog ? genLogTicks(xMin, xMax) : genTicks(xMin, xMax);
  const yTicks = yLog ? genLogTicks(yMin, yMax) : genTicks(yMin, yMax);

  ctx.strokeStyle = THEME.grid;
  ctx.lineWidth = 0.5;
  ctx.setLineDash([]);
  for (const t of xTicks) {
    const px = mapX(t);
    if (px >= mg.l && px <= mg.l + pw) {
      ctx.beginPath(); ctx.moveTo(px, mg.t); ctx.lineTo(px, mg.t + ph); ctx.stroke();
    }
  }
  for (const t of yTicks) {
    const py = mapY(t);
    if (py >= mg.t && py <= mg.t + ph) {
      ctx.beginPath(); ctx.moveTo(mg.l, py); ctx.lineTo(mg.l + pw, py); ctx.stroke();
    }
  }

  // ── Shapes (reference lines) ─────────────────────────────────────
  if (layout?.shapes) {
    for (const sh of layout.shapes) {
      if (sh.type === 'line') {
        const sx0 = sh.xref === 'paper' ? mg.l + (sh.x0 as number) * pw : mapX(sh.x0 as number);
        const sx1 = sh.xref === 'paper' ? mg.l + (sh.x1 as number) * pw : mapX(sh.x1 as number);
        const sy0 = sh.yref === 'paper' ? mg.t + (1 - (sh.y0 as number)) * ph :
          (sh.y0 === 'min' ? mg.t + ph : sh.y0 === 'max' ? mg.t : mapY(sh.y0 as number));
        const sy1 = sh.yref === 'paper' ? mg.t + (1 - (sh.y1 as number)) * ph :
          (sh.y1 === 'min' ? mg.t + ph : sh.y1 === 'max' ? mg.t : mapY(sh.y1 as number));
        ctx.strokeStyle = sh.line?.color || THEME.axis;
        ctx.lineWidth = sh.line?.width || 1;
        setDash(ctx, sh.line?.dash);
        ctx.beginPath(); ctx.moveTo(sx0, sy0); ctx.lineTo(sx1, sy1); ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }

  // ── Draw traces ──────────────────────────────────────────────────
  ctx.save();
  ctx.beginPath();
  ctx.rect(mg.l, mg.t, pw, ph);
  ctx.clip();

  const barTraces = processedTraces.filter(p => p.isHist || p.isBar);
  const barCount = barTraces.length;
  const barMode = layout?.barmode || 'group';
  let barGroupIdx = 0;

  for (let ti = 0; ti < processedTraces.length; ti++) {
    const { trace, xs, ys, isHist, histData, isBar } = processedTraces[ti];
    const color = (typeof trace.marker?.color === 'string' ? trace.marker.color : trace.line?.color) || PALETTE[ti % PALETTE.length];
    const lineWidth = trace.line?.width ?? 1.5;
    const mode = trace.mode || (isHist || isBar ? '' : 'lines');

    // Histogram bars
    if (isHist && histData) {
      ctx.fillStyle = hexToRgba(color, trace.opacity ?? 0.75);
      ctx.strokeStyle = hexToRgba(color, 0.9);
      ctx.lineWidth = 1;
      const gap = layout?.bargap ?? 0.05;
      for (let i = 0; i < histData.counts.length; i++) {
        const left = mapX(histData.edges[i]);
        const right = mapX(histData.edges[i + 1]);
        const top = mapY(histData.counts[i]);
        const bottom = mapY(yLog ? yMin : 0);
        const gapPx = (right - left) * gap;
        ctx.fillRect(left + gapPx, top, right - left - 2 * gapPx, bottom - top);
        ctx.strokeRect(left + gapPx, top, right - left - 2 * gapPx, bottom - top);
      }
      barGroupIdx++;
      continue;
    }

    // Bar chart
    if (isBar) {
      ctx.fillStyle = hexToRgba(color, trace.opacity ?? 0.75);
      ctx.strokeStyle = hexToRgba(color, 0.9);
      ctx.lineWidth = 1;
      const totalBarWidth = pw / Math.max(xs.length, 1) * 0.7;
      const singleBarWidth = barMode === 'group' ? totalBarWidth / Math.max(barCount, 1) : totalBarWidth;
      const offset = barMode === 'group' ? (barGroupIdx - (barCount - 1) / 2) * singleBarWidth : 0;
      for (let i = 0; i < xs.length; i++) {
        const cx = mapX(xs[i]);
        const top = mapY(ys[i]);
        const bottom = mapY(yLog ? yMin : 0);
        ctx.fillRect(cx - singleBarWidth / 2 + offset, top, singleBarWidth * 0.9, bottom - top);
        ctx.strokeRect(cx - singleBarWidth / 2 + offset, top, singleBarWidth * 0.9, bottom - top);
      }
      barGroupIdx++;
      continue;
    }

    // Fill under curve
    if (trace.fill === 'tozeroy' && xs.length > 0) {
      const fillColor = trace.fillcolor || hexToRgba(color, 0.15);
      ctx.fillStyle = fillColor;
      ctx.beginPath();
      ctx.moveTo(mapX(xs[0]), mapY(yLog ? yMin : 0));
      for (let i = 0; i < xs.length; i++) {
        if (isFinite(xs[i]) && isFinite(ys[i])) ctx.lineTo(mapX(xs[i]), mapY(ys[i]));
      }
      ctx.lineTo(mapX(xs[xs.length - 1]), mapY(yLog ? yMin : 0));
      ctx.closePath();
      ctx.fill();
    }

    if (trace.fill === 'tonexty' && ti > 0) {
      const prev = processedTraces[ti - 1];
      const fillColor = trace.fillcolor || hexToRgba(color, 0.15);
      ctx.fillStyle = fillColor;
      ctx.beginPath();
      for (let i = 0; i < xs.length; i++) {
        if (isFinite(xs[i]) && isFinite(ys[i])) {
          if (i === 0) ctx.moveTo(mapX(xs[i]), mapY(ys[i]));
          else ctx.lineTo(mapX(xs[i]), mapY(ys[i]));
        }
      }
      for (let i = prev.xs.length - 1; i >= 0; i--) {
        if (isFinite(prev.xs[i]) && isFinite(prev.ys[i])) {
          ctx.lineTo(mapX(prev.xs[i]), mapY(prev.ys[i]));
        }
      }
      ctx.closePath();
      ctx.fill();
    }

    // Lines
    if (mode.includes('lines')) {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      setDash(ctx, trace.line?.dash);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < xs.length; i++) {
        const px = mapX(xs[i]);
        const py = mapY(ys[i]);
        if (!isFinite(px) || !isFinite(py)) { started = false; continue; }
        if (!started) { ctx.moveTo(px, py); started = true; }
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Markers
    if (mode.includes('markers')) {
      const mSize = trace.marker?.size ?? 4;
      const mColors = Array.isArray(trace.marker?.color) ? trace.marker.color : null;
      for (let i = 0; i < xs.length; i++) {
        const px = mapX(xs[i]);
        const py = mapY(ys[i]);
        if (!isFinite(px) || !isFinite(py)) continue;
        const mc = mColors ? mColors[i] : null;
        ctx.fillStyle = (mc != null && typeof mc === 'string') ? mc : color;
        ctx.beginPath();
        ctx.arc(px, py, mSize / 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Text labels
    if (mode.includes('text') && trace.text) {
      ctx.fillStyle = THEME.textStrong;
      ctx.font = THEME.fontSmall;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      for (let i = 0; i < xs.length; i++) {
        if (trace.text[i]) {
          const px = mapX(xs[i]);
          const py = mapY(ys[i]);
          if (isFinite(px) && isFinite(py)) {
            ctx.fillText(trace.text[i], px, py - 4);
          }
        }
      }
    }
  }

  ctx.restore();

  // ── Axes border ──────────────────────────────────────────────────
  ctx.strokeStyle = THEME.axis;
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  ctx.strokeRect(mg.l, mg.t, pw, ph);

  // ── Tick labels ──────────────────────────────────────────────────
  ctx.fillStyle = THEME.text;
  ctx.font = THEME.fontSmall;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const t of xTicks) {
    const px = mapX(t);
    if (px >= mg.l - 1 && px <= mg.l + pw + 1) {
      ctx.fillText(xLog ? fmtLogNum(t) : fmtNum(t), px, mg.t + ph + 4);
    }
  }
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const t of yTicks) {
    const py = mapY(t);
    if (py >= mg.t - 1 && py <= mg.t + ph + 1) {
      ctx.fillText(yLog ? fmtLogNum(t) : fmtNum(t), mg.l - 6, py);
    }
  }

  // ── Axis titles ──────────────────────────────────────────────────
  ctx.fillStyle = THEME.text;
  ctx.font = THEME.font;
  const xTitle = extractText(layout?.xaxis?.title);
  const yTitle = extractText(layout?.yaxis?.title);
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

  // ── Chart title ──────────────────────────────────────────────────
  const title = extractText(layout?.title);
  if (title) {
    ctx.fillStyle = THEME.textStrong;
    ctx.font = THEME.fontTitle;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(title, mg.l + pw / 2, mg.t - 8);
  }

  // ── Legend ────────────────────────────────────────────────────────
  const showLegend = layout?.showlegend ?? data.some(d => d.name);
  const legendItems = data.filter(d => d.name && (d.showlegend !== false));
  if (showLegend && legendItems.length > 0) {
    ctx.font = THEME.fontSmall;
    const legendX = mg.l + pw - 8;
    let legendY = mg.t + 8;
    for (let i = 0; i < legendItems.length; i++) {
      const tr = legendItems[i];
      const c = (typeof tr.marker?.color === 'string' ? tr.marker.color : tr.line?.color) || PALETTE[i % PALETTE.length];
      const textW = ctx.measureText(tr.name!).width;
      // Background
      ctx.fillStyle = THEME.legendBg;
      ctx.fillRect(legendX - textW - 24, legendY - 2, textW + 28, 16);
      // Color line
      ctx.strokeStyle = c;
      ctx.lineWidth = 2;
      setDash(ctx, tr.line?.dash);
      ctx.beginPath();
      ctx.moveTo(legendX - textW - 20, legendY + 6);
      ctx.lineTo(legendX - textW - 6, legendY + 6);
      ctx.stroke();
      ctx.setLineDash([]);
      // Label
      ctx.fillStyle = THEME.text;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(tr.name!, legendX - textW - 2, legendY + 6);
      legendY += 18;
    }
  }
}

// ── React Component ────────────────────────────────────────────────────

export function CanvasChart({ data, layout, style }: CanvasChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const themeKey = useThemeChangeKey();
  const isFullscreen = useSimulationFullscreen();
  const insideMain = useIsInsideMain();

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const redraw = () => {
      THEME = getCanvasTheme();
      const width = container.clientWidth;
      const height = isFullscreen && container.clientHeight > 0
        ? container.clientHeight
        : parseInt(String(style?.height || 320), 10);
      const dpr = window.devicePixelRatio || 1;

      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.scale(dpr, dpr);

      // Clear
      ctx.clearRect(0, 0, width, height);
      draw(ctx, width, height, data, layout);
    };

    redraw();

    const ro = new ResizeObserver(redraw);
    ro.observe(container);
    return () => ro.disconnect();
  }, [data, layout, style, themeKey, isFullscreen]);

  return (
    <div
      ref={containerRef}
      data-fs-role={isFullscreen && !insideMain ? 'chart' : undefined}
      style={
        isFullscreen && insideMain === 'main'
          ? { width: '100%', height: '100%' }
          : isFullscreen && !insideMain
            ? undefined
            : { width: style?.width || '100%' }
      }
    >
      <canvas ref={canvasRef} style={{ display: 'block', borderRadius: '4px' }} />
    </div>
  );
}
