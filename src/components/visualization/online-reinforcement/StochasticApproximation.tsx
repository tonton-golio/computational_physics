'use client';

import React, { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';
import { Slider } from '@/components/ui/slider';
import { usePlotlyTheme } from '@/lib/plotly-theme';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id: string;
}

function mulberry32(a: number) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export default function StochasticApproximation({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [steps, setSteps] = useState(300);
  const [alpha, setAlpha] = useState(0.03);
  const [noiseScale, setNoiseScale] = useState(0.1);
  const [seed, setSeed] = useState(7);
  const { mergeLayout } = usePlotlyTheme();

  const { xs, ys } = useMemo(() => {
    const rng = mulberry32(seed);
    const xVals: number[] = [0.6];
    const yVals: number[] = [Math.abs(xVals[0] * xVals[0] - 1)];
    for (let k = 1; k <= steps; k++) {
      const prev = xVals[k - 1];
      const noise = (rng() - 0.5) * 2 * noiseScale;
      const gradientProxy = (prev * prev - 1) + noise;
      const next = prev - alpha * gradientProxy;
      xVals.push(next);
      yVals.push(Math.abs(next * next - 1));
    }
    return { xs: xVals, ys: yVals };
  }, [steps, alpha, noiseScale, seed]);

  return (
    <div className="w-full rounded-lg bg-[var(--surface-1)] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-3 text-[var(--text-strong)]">Robbins-Monro Stochastic Approximation</h3>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Steps: {steps}</label>
          <Slider value={[steps]} onValueChange={([v]) => setSteps(v)} min={50} max={1000} step={10} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Step size: {alpha.toFixed(3)}</label>
          <Slider value={[alpha]} onValueChange={([v]) => setAlpha(v)} min={0.005} max={0.1} step={0.005} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Noise: {noiseScale.toFixed(2)}</label>
          <Slider value={[noiseScale]} onValueChange={([v]) => setNoiseScale(v)} min={0} max={0.5} step={0.01} />
        </div>
        <button onClick={() => setSeed((s) => s + 1)} className="rounded bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white text-sm px-3 py-2">Re-sample</button>
      </div>
      <Plot
        data={[
          { x: Array.from({ length: xs.length }, (_, i) => i), y: xs, type: 'scatter', mode: 'lines', name: 'x_k', line: { color: '#60a5fa', width: 2 } },
          { x: Array.from({ length: ys.length }, (_, i) => i), y: ys.map((v) => Math.log10(v + 1e-6)), type: 'scatter', mode: 'lines', name: 'log10 residual', line: { color: '#facc15', width: 2 } },
        ]}
        layout={mergeLayout({
          title: { text: 'Convergence behavior under noisy updates' },
          xaxis: { title: { text: 'iteration k' } },
          yaxis: { title: { text: 'state / log residual' } },
          height: 420,
          margin: { t: 40, b: 60, l: 60, r: 60 },
        })}
        config={{ displayModeBar: false }}
        style={{ width: '100%' }}
      />
    </div>
  );
}
