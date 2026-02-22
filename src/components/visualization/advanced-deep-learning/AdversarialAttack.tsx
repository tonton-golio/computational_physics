"use client";

import { useState, useMemo, useRef, useEffect } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart, type ChartTrace } from '@/components/ui/canvas-chart';
import { SimulationMain } from '@/components/ui/simulation-main';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationAux, SimulationLabel } from '@/components/ui/simulation-panel';
import type { SimulationComponentProps } from '@/shared/types/simulation';

const SZ = 12, CELL = 20, NC = 4;
const CNAMES = ['Circle', 'Cross', 'Line', 'Square'];
const CCOLS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'];

function rng(seed: number) {
  return () => { let t = (seed += 0x6d2b79f5); t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

function genPattern(type: number): number[][] {
  const img = Array.from({ length: SZ }, () => Array(SZ).fill(0));
  const c = SZ >> 1, r = SZ / 3 | 0;
  if (type === 0) for (let y = 0; y < SZ; y++) for (let x = 0; x < SZ; x++) {
    if (Math.abs(Math.hypot(x - c, y - c) - r) < 1.2) img[y][x] = 1;
  } else if (type === 1) for (let i = 2; i < SZ - 2; i++) { img[c][i] = 1; img[i][c] = 1; }
  else if (type === 2) for (let i = 1; i < SZ - 1; i++) { img[i][i] = 1; if (i + 1 < SZ) img[i][i + 1] = .5; }
  else for (let i = 2; i < SZ - 2; i++) { img[2][i] = 1; img[SZ - 3][i] = 1; img[i][2] = 1; img[i][SZ - 3] = 1; }
  return img;
}

function computeGrad(img: number[][], cls: number): number[][] {
  const r = rng(cls * 1000 + 42);
  return Array.from({ length: SZ }, (_, y) =>
    Array.from({ length: SZ }, (_, x) =>
      (img[y][x] > 0.3 ? -0.8 : 0.3) + Math.sin(x * 1.2 + cls) * Math.cos(y * 0.8 + cls) * 0.5 + (r() - 0.5) * 0.4
    )
  );
}

function confidence(_img: number[][], cls: number, eps: number): number[] {
  const logits = [0.5, 0.5, 0.5, 0.5];
  logits[cls] = 4.0;
  const tgt = (cls + 1) % NC, s = eps * 30;
  logits[cls] -= s * 0.8; logits[tgt] += s * 0.6;
  for (let c = 0; c < NC; c++) if (c !== cls && c !== tgt) logits[c] += s * 0.1;
  const mx = Math.max(...logits), exps = logits.map(l => Math.exp(l - mx));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function drawImg(ctx: CanvasRenderingContext2D, img: number[][], ox: number, oy: number, label: string) {
  let lo = Infinity, hi = -Infinity;
  for (const row of img) for (const v of row) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const rng = Math.max(hi - lo, 0.01);
  for (let y = 0; y < SZ; y++) for (let x = 0; x < SZ; x++) {
    const b = ((img[y][x] - lo) / rng * 220) | 0;
    ctx.fillStyle = `rgb(${b},${b},${b + 20})`; ctx.fillRect(ox + x * CELL, oy + y * CELL, CELL - 1, CELL - 1);
  }
  ctx.fillStyle = '#a0aec0'; ctx.font = '11px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  ctx.fillText(label, ox + SZ * CELL / 2, oy + SZ * CELL + 4);
}

export default function AdversarialAttack({}: SimulationComponentProps) {
  const [epsilon, setEpsilon] = useState(0.1);
  const [patType, setPatType] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const original = useMemo(() => genPattern(patType), [patType]);
  const grad = useMemo(() => computeGrad(original, patType), [original, patType]);
  const pert = useMemo(() => grad.map(r => r.map(g => epsilon * (g > 0 ? 1 : -1))), [grad, epsilon]);
  const adv = useMemo(() => original.map((r, y) => r.map((v, x) => Math.max(0, Math.min(1, v + pert[y][x])))), [original, pert]);
  const origConf = useMemo(() => confidence(original, patType, 0), [original, patType]);
  const advConf = useMemo(() => confidence(adv, patType, epsilon), [adv, patType, epsilon]);

  useEffect(() => {
    const canvas = canvasRef.current, container = containerRef.current;
    if (!canvas || !container) return;
    const redraw = () => {
      const pw = SZ * CELL, gap = 20, plus = 28;
      const w = pw * 3 + gap * 2 + plus * 2, h = pw + 28;
      const dpr = devicePixelRatio || 1;
      canvas.width = w * dpr; canvas.height = h * dpr;
      canvas.style.width = `${w}px`; canvas.style.height = `${h}px`;
      const ctx = canvas.getContext('2d'); if (!ctx) return;
      ctx.scale(dpr, dpr); ctx.clearRect(0, 0, w, h);
      drawImg(ctx, original, 0, 0, 'Original');
      ctx.fillStyle = '#a0aec0'; ctx.font = '18px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('+', pw + plus / 2, pw / 2);
      drawImg(ctx, pert.map(r => r.map(v => v * 5 + 0.5)), pw + plus, 0, `Perturbation (eps=${epsilon.toFixed(2)})`);
      ctx.fillText('=', pw * 2 + plus + gap + plus / 2, pw / 2);
      drawImg(ctx, adv, pw * 2 + plus * 2 + gap, 0, 'Adversarial');
    };
    redraw();
    const ro = new ResizeObserver(redraw); ro.observe(container);
    return () => ro.disconnect();
  }, [original, pert, adv, epsilon]);

  const barTraces: ChartTrace[] = [
    { x: CNAMES, y: origConf, type: 'bar', marker: { color: CCOLS.map(c => c + '99') }, name: 'Original' },
    { x: CNAMES, y: advConf, type: 'bar', marker: { color: CCOLS }, name: 'Adversarial' },
  ];

  const sweepTraces: ChartTrace[] = useMemo(() => {
    const eps = Array.from({ length: 50 }, (_, i) => i * 0.01);
    const s: number[][] = Array.from({ length: NC }, () => []);
    for (const e of eps) { const c = confidence(original, patType, e); for (let i = 0; i < NC; i++) s[i].push(c[i]); }
    return s.map((ys, c) => ({ x: eps, y: ys, type: 'scatter' as const, mode: 'lines' as const,
      line: { color: CCOLS[c], width: c === patType ? 3 : 1.5 }, name: CNAMES[c] }));
  }, [original, patType]);

  const pred = advConf.indexOf(Math.max(...advConf));

  return (
    <SimulationPanel title="Adversarial Attack (FGSM)">
      <SimulationConfig>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">Perturbation epsilon: {epsilon.toFixed(2)}</SimulationLabel>
            <Slider min={0} max={0.5} step={0.01} value={[epsilon]} onValueChange={([v]) => setEpsilon(v)} />
          </div>
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">Input pattern: {CNAMES[patType]}</SimulationLabel>
            <Slider min={0} max={NC - 1} step={1} value={[patType]} onValueChange={([v]) => setPatType(v)} />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain scaleMode="contain" className="flex justify-center mb-4 overflow-x-auto">
        <div ref={containerRef}><canvas ref={canvasRef} style={{ display: 'block', borderRadius: '4px' }} /></div>
      </SimulationMain>
      <SimulationAux>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <CanvasChart data={barTraces} layout={{ title: { text: 'Classifier confidence' },
            yaxis: { title: { text: 'Probability' }, range: [0, 1] }, margin: { t: 40, b: 50, l: 60, r: 20 },
            barmode: 'group', bargap: 0.3 }} style={{ width: '100%', height: 320 }} />
          <CanvasChart data={[...sweepTraces, { x: [epsilon, epsilon], y: [0, 1], type: 'scatter', mode: 'lines',
            line: { color: '#f59e0b', width: 1.5, dash: 'dash' }, showlegend: false }]}
            layout={{ title: { text: 'Confidence vs. epsilon' }, xaxis: { title: { text: 'Epsilon' } },
              yaxis: { title: { text: 'Probability' }, range: [0, 1] }, margin: { t: 40, b: 50, l: 60, r: 20 } }}
            style={{ width: '100%', height: 320 }} />
        </div>
      </SimulationAux>
      {pred !== patType && (
        <SimulationResults>
          <div className="p-3 bg-red-900/20 border border-red-500/30 rounded text-sm text-red-300">
            Classification flipped: {CNAMES[patType]} {'->'} {CNAMES[pred]} at epsilon = {epsilon.toFixed(2)}
          </div>
        </SimulationResults>
      )}
      <div className="mt-4 p-3 bg-[var(--surface-2)] rounded text-sm text-[var(--text-muted)]">
        FGSM adds a perturbation of magnitude epsilon in the direction that maximally increases the loss.
        Even tiny epsilon values can flip the classifier prediction. The right panel shows how each class
        probability changes as epsilon increases, revealing the decision boundary fragility.
      </div>
    </SimulationPanel>
  );
}
