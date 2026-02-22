"use client";

import { useState, useMemo, useRef, useEffect } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart, type ChartTrace } from '@/components/ui/canvas-chart';
import { SimulationMain } from '@/components/ui/simulation-main';
import { SimulationPanel, SimulationConfig, SimulationAux, SimulationLabel } from '@/components/ui/simulation-panel';
import type { SimulationComponentProps } from '@/shared/types/simulation';

const FS = 5, NF = 6, CELL = 26;
const NAMES = ['Horiz. edge', 'Vert. edge', 'Diag. edge', 'Gaussian', 'Laplacian', 'Gabor'];
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

function rng(seed: number) {
  return () => { let t = (seed += 0x6d2b79f5); t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

const mkF = (fn: (r: number, c: number) => number) =>
  Array.from({ length: FS }, (_, r) => Array.from({ length: FS }, (_, c) => fn(r, c)));

const TARGETS = [
  mkF((r) => r < 2 ? -1 : r > 2 ? 1 : 0),
  mkF((_, c) => c < 2 ? -1 : c > 2 ? 1 : 0),
  mkF((r, c) => { const d = c - r; return d < -1 ? -1 : d > 1 ? 1 : 0; }),
  mkF((r, c) => Math.exp(-((r - 2) ** 2 + (c - 2) ** 2) / 2)),
  mkF((r, c) => { const d = Math.abs(r - 2) + Math.abs(c - 2); return d <= 1 ? 1 : d === 2 ? -0.5 : -0.25; }),
  mkF((r, c) => Math.cos(2 * (c - 2)) * Math.exp(-((c - 2) ** 2 + (r - 2) ** 2) / 3)),
];

function interp(a: number[][], b: number[][], t: number) {
  return a.map((row, r) => row.map((v, c) => v * (1 - t) + b[r][c] * t));
}

function mse(a: number[][], b: number[][]) {
  let s = 0; for (let r = 0; r < FS; r++) for (let c = 0; c < FS; c++) s += (a[r][c] - b[r][c]) ** 2;
  return s / (FS * FS * 4);
}

export default function FilterEvolution({}: SimulationComponentProps) {
  const [epoch, setEpoch] = useState(0);
  const [selFilter, setSelFilter] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const maxEp = 100;

  const initFilters = useMemo(() =>
    Array.from({ length: NF }, (_, f) => {
      const r = rng(f * 1000 + 7);
      return mkF(() => (r() - 0.5) * 2);
    }), []);

  const progress = useMemo(() => 1 - Math.pow(1 - Math.min(1, epoch / maxEp), 2.5), [epoch]);
  const filters = useMemo(() => initFilters.map((init, f) => interp(init, TARGETS[f], progress)), [initFilters, progress]);
  const conv = useMemo(() => filters.map((f, i) => 1 - mse(f, TARGETS[i])), [filters]);

  useEffect(() => {
    const canvas = canvasRef.current, container = containerRef.current;
    if (!canvas || !container) return;
    const redraw = () => {
      const gap = 10, lbl = 16, cols = 3, rows = Math.ceil(NF / cols);
      const fw = FS * CELL, tw = cols * fw + (cols - 1) * gap, th = rows * (fw + lbl + gap);
      const dpr = devicePixelRatio || 1;
      canvas.width = tw * dpr; canvas.height = th * dpr;
      canvas.style.width = `${tw}px`; canvas.style.height = `${th}px`;
      const ctx = canvas.getContext('2d'); if (!ctx) return;
      ctx.scale(dpr, dpr); ctx.clearRect(0, 0, tw, th);

      for (let f = 0; f < NF; f++) {
        const col = f % cols, row = (f / cols) | 0;
        const ox = col * (fw + gap), oy = row * (fw + lbl + gap);
        const filter = filters[f];
        let rng = 0; for (const r of filter) for (const v of r) rng = Math.max(rng, Math.abs(v));
        rng = rng || 1;
        for (let r = 0; r < FS; r++) for (let c = 0; c < FS; c++) {
          const val = filter[r][c] / rng;
          const x = ox + c * CELL, y = oy + r * CELL;
          ctx.fillStyle = val > 0 ? `rgb(${59 + (val * 200 | 0)},${130 + (val * 60 | 0)},246)` : `rgb(239,${68 + (-val * 60 | 0)},${68 + (-val * 200 | 0)})`;
          ctx.fillRect(x, y, CELL - 1, CELL - 1);
          ctx.fillStyle = Math.abs(val) > 0.5 ? '#fff' : '#ccc';
          ctx.font = '9px monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(filter[r][c].toFixed(1), x + CELL / 2, y + CELL / 2);
        }
        if (f === selFilter) { ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 2; ctx.strokeRect(ox - 1, oy - 1, fw + 2, fw + 2); }
        ctx.fillStyle = '#a0aec0'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        ctx.fillText(NAMES[f], ox + fw / 2, oy + fw + 2);
      }
    };
    redraw();
    const ro = new ResizeObserver(redraw); ro.observe(container);
    return () => ro.disconnect();
  }, [filters, selFilter]);

  const chartTraces: ChartTrace[] = useMemo(() => {
    const eps: number[] = [];
    const data: number[][] = Array.from({ length: NF }, () => []);
    for (let e = 0; e <= maxEp; e += 2) {
      eps.push(e);
      const p = 1 - Math.pow(1 - Math.min(1, e / maxEp), 2.5);
      for (let f = 0; f < NF; f++) data[f].push(1 - mse(interp(initFilters[f], TARGETS[f], p), TARGETS[f]));
    }
    return data.map((ys, f) => ({ x: eps, y: ys, type: 'scatter' as const, mode: 'lines' as const,
      line: { color: COLORS[f], width: f === selFilter ? 3 : 1.5, dash: f === selFilter ? 'solid' as const : 'dash' as const },
      name: NAMES[f] }));
  }, [initFilters, selFilter]);

  return (
    <SimulationPanel title="Filter Evolution During Training">
      <SimulationConfig>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">Training epoch: {epoch}</SimulationLabel>
            <Slider min={0} max={maxEp} step={1} value={[epoch]} onValueChange={([v]) => setEpoch(v)} />
          </div>
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">Highlight: {NAMES[selFilter]}</SimulationLabel>
            <Slider min={0} max={NF - 1} step={1} value={[selFilter]} onValueChange={([v]) => setSelFilter(v)} />
          </div>
        </div>
      </SimulationConfig>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
        <SimulationMain scaleMode="contain" className="flex flex-col items-center">
          <p className="text-sm text-[var(--text-muted)] mb-2 font-semibold">
            {FS}x{FS} learned filters ({(conv[selFilter] * 100).toFixed(0)}% converged)
          </p>
          <div ref={containerRef}><canvas ref={canvasRef} style={{ display: 'block', borderRadius: '4px' }} /></div>
          <div className="flex gap-4 mt-3 text-xs text-[var(--text-muted)]">
            <span><span className="inline-block w-3 h-3 rounded-sm bg-blue-500 mr-1 align-middle" /> Positive</span>
            <span><span className="inline-block w-3 h-3 rounded-sm bg-red-500 mr-1 align-middle" /> Negative</span>
          </div>
        </SimulationMain>
        <SimulationAux>
          <CanvasChart data={chartTraces} layout={{ title: { text: 'Filter convergence' },
            xaxis: { title: { text: 'Epoch' } }, yaxis: { title: { text: 'Convergence' }, range: [0, 1.05] },
            margin: { t: 40, b: 50, l: 60, r: 20 },
            shapes: [{ type: 'line', x0: epoch, x1: epoch, y0: 0, y1: 1.05,
              line: { color: '#f59e0b', width: 1.5, dash: 'dash' } }] }}
            style={{ width: '100%', height: 370 }} />
        </SimulationAux>
      </div>
      <div className="mt-4 p-3 bg-[var(--surface-2)] rounded text-sm text-[var(--text-muted)]">
        Filters start as random noise and gradually organize into structured patterns during training.
        Early layers typically learn edge detectors, blob detectors, and Gabor-like texture filters.
        Drag the epoch slider to watch the evolution from noise to organized feature detectors.
      </div>
    </SimulationPanel>
  );
}
