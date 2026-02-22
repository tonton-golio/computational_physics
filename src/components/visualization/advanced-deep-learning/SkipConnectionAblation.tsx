"use client";

import { useState, useMemo } from 'react';
import { CanvasChart, type ChartTrace, type ChartLayout } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationLabel, SimulationCheckbox } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

function rng(seed: number) {
  return () => { let t = (seed += 0x6d2b79f5); t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

function simulate(depth: number, skip: boolean, lr: number) {
  const r = rng(depth * 100 + (skip ? 1 : 0)), N = 100;
  const train: number[] = [], val: number[] = [], grad: number[] = [];
  const vf = skip ? 1 : Math.pow(0.85, depth), eLR = lr * vf;
  let loss = 2.5, vLoss = 2.6;

  for (let e = 0; e < N; e++) {
    const t = e / N, n = (r() - 0.5) * 0.08;
    if (skip) {
      const dr = 3.5 * Math.min(1, eLR * 50);
      loss = 0.15 + 2.35 * Math.exp(-dr * t) + n * 0.5;
      vLoss = loss + 0.05 + 0.1 * t * depth / 50 + n * 0.3;
    } else if (depth <= 10) {
      const dr = 2.5 * Math.min(1, eLR * 50);
      loss = 0.3 + 2.2 * Math.exp(-dr * t) + n * 0.5;
      vLoss = loss + 0.1 + n * 0.3;
    } else {
      const stall = 2.5 - 0.7 * (1 - Math.pow(0.9, depth - 10));
      const sd = 0.3 * vf;
      loss = stall - sd * t + n; vLoss = stall - sd * t * 0.5 + Math.abs(n) * 1.5;
    }
    train.push(Math.max(0.05, loss)); val.push(Math.max(0.1, vLoss));
    const gn = skip ? 0.8 + 0.4 * Math.exp(-t) + n * 0.1 : vf * (0.5 + 0.5 * Math.exp(-t * 0.5)) + n * 0.05;
    grad.push(Math.max(1e-6, gn));
  }
  return { train, val, grad };
}

export default function SkipConnectionAblation({}: SimulationComponentProps) {
  const [depth, setDepth] = useState(30);
  const [lr, setLr] = useState(0.01);
  const [showSkip, setShowSkip] = useState(true);
  const [showNoSkip, setShowNoSkip] = useState(true);

  const withSkip = useMemo(() => simulate(depth, true, lr), [depth, lr]);
  const noSkip = useMemo(() => simulate(depth, false, lr), [depth, lr]);
  const epochs = Array.from({ length: 100 }, (_, i) => i + 1);

  const lossTraces: ChartTrace[] = useMemo(() => {
    const t: ChartTrace[] = [];
    if (showSkip) {
      t.push({ x: epochs, y: withSkip.train, type: 'scatter', mode: 'lines', line: { color: '#3b82f6', width: 2.5 }, name: 'Train (skip)' });
      t.push({ x: epochs, y: withSkip.val, type: 'scatter', mode: 'lines', line: { color: '#3b82f6', width: 1.5, dash: 'dash' }, name: 'Val (skip)' });
    }
    if (showNoSkip) {
      t.push({ x: epochs, y: noSkip.train, type: 'scatter', mode: 'lines', line: { color: '#ef4444', width: 2.5 }, name: 'Train (no skip)' });
      t.push({ x: epochs, y: noSkip.val, type: 'scatter', mode: 'lines', line: { color: '#ef4444', width: 1.5, dash: 'dash' }, name: 'Val (no skip)' });
    }
    return t;
  }, [epochs, withSkip, noSkip, showSkip, showNoSkip]);

  const gradTraces: ChartTrace[] = useMemo(() => {
    const t: ChartTrace[] = [];
    if (showSkip) t.push({ x: epochs, y: withSkip.grad, type: 'scatter', mode: 'lines', line: { color: '#3b82f6', width: 2 }, name: 'With skip' });
    if (showNoSkip) t.push({ x: epochs, y: noSkip.grad, type: 'scatter', mode: 'lines', line: { color: '#ef4444', width: 2 }, name: 'Without skip' });
    return t;
  }, [epochs, withSkip, noSkip, showSkip, showNoSkip]);

  const lossLayout: ChartLayout = { title: { text: 'Training & validation loss' },
    xaxis: { title: { text: 'Epoch' } }, yaxis: { title: { text: 'Loss' } }, margin: { t: 40, b: 50, l: 60, r: 20 } };
  const gradLayout: ChartLayout = { title: { text: 'Gradient norm (layer 1)' },
    xaxis: { title: { text: 'Epoch' } }, yaxis: { title: { text: 'Gradient norm' }, type: 'log' }, margin: { t: 40, b: 50, l: 60, r: 20 } };

  return (
    <SimulationPanel title="Skip Connection Ablation">
      <SimulationSettings>
        <div className="flex gap-4">
          <SimulationCheckbox checked={showSkip} onChange={setShowSkip} label="With skip connections" style={{ color: showSkip ? '#3b82f6' : '#666' }} />
          <SimulationCheckbox checked={showNoSkip} onChange={setShowNoSkip} label="Without skip connections" style={{ color: showNoSkip ? '#ef4444' : '#666' }} />
        </div>
      </SimulationSettings>
      <SimulationConfig>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">Network depth: {depth} layers</SimulationLabel>
            <Slider min={4} max={80} step={2} value={[depth]} onValueChange={([v]) => setDepth(v)} />
          </div>
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">Learning rate: {lr.toFixed(3)}</SimulationLabel>
            <Slider min={0.001} max={0.1} step={0.001} value={[lr]} onValueChange={([v]) => setLr(v)} />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <CanvasChart data={lossTraces} layout={lossLayout} style={{ width: '100%', height: 380 }} />
          <CanvasChart data={gradTraces} layout={gradLayout} style={{ width: '100%', height: 380 }} />
        </div>
      </SimulationMain>
      <div className="mt-4 p-3 bg-[var(--surface-2)] rounded text-sm text-[var(--text-muted)]">
        {depth > 20
          ? `At ${depth} layers, the network without skip connections shows severe gradient vanishing — the gradient norm is orders of magnitude smaller, causing training to stall. With residual connections, gradients flow freely through the identity shortcut.`
          : `At ${depth} layers, both networks train reasonably. Increase depth beyond 20 to see the dramatic effect: without skip connections, the gradient signal decays exponentially through each layer.`}
      </div>
    </SimulationPanel>
  );
}
