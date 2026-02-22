"use client";

import { useState, useMemo } from 'react';
import { gaussianPair } from '@/lib/math';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

function gammaRandom(shape: number): number {
  // Marsaglia and Tsang's method for shape >= 1
  if (shape < 1) {
    return gammaRandom(shape + 1) * Math.pow(Math.random(), 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number, v: number;
    do {
      x = gaussianPair(Math.random)[0];
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

export default function WhichAverage({}: SimulationComponentProps) {
  const [skewness, setSkewness] = useState(0);
  const [numSamples, setNumSamples] = useState(5000);

  const { bins, counts, mean, median, mode, maxCount } = useMemo(() => {
    const N = numSamples;
    const samples: number[] = [];

    if (skewness < 0.1) {
      // Normal distribution — truly symmetric
      for (let i = 0; i < N; i++) {
        samples.push(5 + gaussianPair(Math.random)[0]);
      }
    } else {
      // Gamma with target skewness: gamma_skew = 2/sqrt(alpha) → alpha = 4/skew²
      const alpha = 4 / (skewness * skewness);
      for (let i = 0; i < N; i++) {
        // Standardize to mean 5, std 1 so chart stays comparable across skewness values
        samples.push((gammaRandom(alpha) - alpha) / Math.sqrt(alpha) + 5);
      }
    }
    const sorted = [...samples].sort((a, b) => a - b);
    const mn = sorted.reduce((s, v) => s + v, 0) / N;
    const med = N % 2 === 0 ? (sorted[N / 2 - 1] + sorted[N / 2]) / 2 : sorted[Math.floor(N / 2)];

    const nbins = 60;
    const lo = sorted[0];
    const hi = sorted[N - 1];
    const step = (hi - lo) / nbins;
    const bns: number[] = [];
    const cts = new Array(nbins).fill(0);
    for (let i = 0; i < nbins; i++) bns.push(lo + (i + 0.5) * step);
    for (const v of samples) {
      let idx = Math.floor((v - lo) / step);
      if (idx >= nbins) idx = nbins - 1;
      if (idx < 0) idx = 0;
      cts[idx]++;
    }
    let modeIdx = 0;
    for (let i = 1; i < nbins; i++) {
      if (cts[i] > cts[modeIdx]) modeIdx = i;
    }
    const mx = Math.max(...cts);
    return { bins: bns, counts: cts, mean: mn, median: med, mode: bns[modeIdx], maxCount: mx };
  }, [skewness, numSamples]);

  return (
    <SimulationPanel title="Which Average? Mean vs Median vs Mode">
      <SimulationConfig>
        <div>
          <SimulationLabel>Skewness: {skewness.toFixed(1)}</SimulationLabel>
          <Slider value={[skewness]} onValueChange={([v]) => setSkewness(v)} min={0} max={5} step={0.1} />
        </div>
        <div>
          <SimulationLabel>Samples: {numSamples}</SimulationLabel>
          <Slider value={[numSamples]} onValueChange={([v]) => setNumSamples(v)} min={100} max={20000} step={100} />
        </div>
      </SimulationConfig>
      <SimulationMain>
      <CanvasChart
        data={[
          { x: bins, y: counts, type: 'bar', marker: { color: '#3b82f6' }, opacity: 0.6, name: 'Histogram' },
        ]}
        layout={{
          height: 400,
          xaxis: { title: { text: 'Value' } },
          yaxis: { title: { text: 'Count' }, range: [0, maxCount * 1.15] },
          shapes: [
            { type: 'line', x0: mean, x1: mean, y0: 0, y1: maxCount * 1.1, line: { color: '#ef4444', width: 2.5, dash: 'solid' } },
            { type: 'line', x0: median, x1: median, y0: 0, y1: maxCount * 1.1, line: { color: '#10b981', width: 2.5, dash: 'dash' } },
            { type: 'line', x0: mode, x1: mode, y0: 0, y1: maxCount * 1.1, line: { color: '#8b5cf6', width: 2.5, dash: 'dot' } },
          ],
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>
      <SimulationResults>
        <div className="text-sm text-[var(--text-muted)] flex flex-wrap gap-4">
          <span><span className="inline-block w-3 h-3 rounded mr-1" style={{ background: '#ef4444' }} /> Mean: {mean.toFixed(2)}</span>
          <span><span className="inline-block w-3 h-3 rounded mr-1" style={{ background: '#10b981' }} /> Median: {median.toFixed(2)}</span>
          <span><span className="inline-block w-3 h-3 rounded mr-1" style={{ background: '#8b5cf6' }} /> Mode: {mode.toFixed(2)}</span>
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
