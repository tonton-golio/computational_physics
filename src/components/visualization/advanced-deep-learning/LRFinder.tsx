"use client";

import { useState, useMemo } from 'react';
import { CanvasChart, type ChartTrace, type ChartLayout } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

function rng(seed: number) {
  return () => { let t = (seed += 0x6d2b79f5); t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

function simulateLRFinder(batchSize: number, complexity: number) {
  const r = rng(42), steps = 200, lrMin = 1e-6, lrMax = 10;
  const lrs: number[] = [], losses: number[] = [], smoothed: number[] = [];
  const noise = 2 / Math.sqrt(batchSize);
  const optLog = -2.5 + (complexity - 1) * 0.3, divLog = optLog + 1.8;
  let sm = 0; const beta = 0.98;

  for (let i = 0; i < steps; i++) {
    const frac = i / (steps - 1);
    const logLR = Math.log10(lrMin) + frac * (Math.log10(lrMax) - Math.log10(lrMin));
    const lr = Math.pow(10, logLR);
    lrs.push(lr);
    const base = 2.5 - 1.5 / (1 + Math.exp(-8 * (logLR - (optLog - 1))));
    const div = Math.max(0, Math.exp(3 * (logLR - divLog)));
    const loss = Math.max(0.05, base + div + noise * (r() - 0.5));
    losses.push(loss);
    sm = beta * sm + (1 - beta) * loss;
    smoothed.push(sm / (1 - Math.pow(beta, i + 1)));
  }
  return { lrs, losses, smoothed };
}

export default function LRFinder({}: SimulationComponentProps) {
  const [batchSize, setBatchSize] = useState(64);
  const [complexity, setComplexity] = useState(1.0);

  const { lrs, losses, smoothed } = useMemo(
    () => simulateLRFinder(batchSize, complexity), [batchSize, complexity]);

  const minIdx = useMemo(() => {
    let best = 0, bestVal = Infinity;
    for (let i = 10; i < smoothed.length - 10; i++) if (smoothed[i] < bestVal) { bestVal = smoothed[i]; best = i; }
    return best;
  }, [smoothed]);

  const traces: ChartTrace[] = [
    { x: lrs, y: losses, type: 'scatter', mode: 'lines', line: { color: '#6366f1', width: 1 }, name: 'Raw loss', opacity: 0.4 },
    { x: lrs, y: smoothed, type: 'scatter', mode: 'lines', line: { color: '#3b82f6', width: 2.5 }, name: 'Smoothed loss' },
    { x: [lrs[minIdx]], y: [smoothed[minIdx]], type: 'scatter', mode: 'markers', marker: { size: 10, color: '#10b981' }, name: 'Suggested LR' },
  ];

  const layout: ChartLayout = {
    xaxis: { title: { text: 'Learning Rate' }, type: 'log' },
    yaxis: { title: { text: 'Loss' } },
    margin: { t: 30, b: 50, l: 60, r: 30 },
    shapes: [{ type: 'line', x0: lrs[minIdx], x1: lrs[minIdx], y0: 0, y1: 4,
      line: { color: '#10b981', width: 1.5, dash: 'dash' } }],
  };

  return (
    <SimulationPanel title="Learning Rate Range Test">
      <SimulationConfig>
        <div className="space-y-4">
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">Batch size: {batchSize}</SimulationLabel>
            <Slider min={8} max={512} step={8} value={[batchSize]} onValueChange={([v]) => setBatchSize(v)} />
          </div>
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">Model complexity: {complexity.toFixed(1)}</SimulationLabel>
            <Slider min={0.5} max={3.0} step={0.1} value={[complexity]} onValueChange={([v]) => setComplexity(v)} />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
        <CanvasChart data={traces} layout={layout} style={{ width: '100%', height: 420 }} />
      </SimulationMain>
      <SimulationResults>
        <div className="p-3 bg-[var(--surface-2)] rounded text-xs text-[var(--text-muted)] space-y-2">
          <p><span className="text-amber-400 font-semibold">Too small:</span> Loss is flat — the learning rate makes no meaningful progress.</p>
          <p><span className="text-green-400 font-semibold">Just right:</span> Loss decreases steadily. Suggested LR: {lrs[minIdx]?.toExponential(1)}.</p>
          <p><span className="text-red-400 font-semibold">Too large:</span> Loss diverges — updates overshoot and destabilize training.</p>
          <p className="mt-2">Larger batch sizes reduce noise. Higher complexity shifts the optimal range.</p>
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
