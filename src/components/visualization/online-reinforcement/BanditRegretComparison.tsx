'use client';

import React, { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export default function BanditRegretComparison({ id }: SimulationComponentProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [horizon, setHorizon] = useState(1000);
  const t = useMemo(() => Array.from({ length: horizon }, (_, i) => i + 1), [horizon]);
  const ucb = useMemo(() => t.map((x) => 0.8 * Math.sqrt(x * Math.log(x + 1))), [t]);
  const exp3 = useMemo(() => t.map((x) => 1.2 * Math.sqrt(4 * x * Math.log(4))), [t]);
  const epsGreedy = useMemo(() => t.map((x) => 0.06 * x + 8 * Math.sqrt(x)), [t]);
  const random = useMemo(() => t.map((x) => 0.22 * x), [t]);

  return (
    <div className="w-full rounded-lg bg-[var(--surface-1)] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-3 text-[var(--text-strong)]">Bandit Regret Comparison</h3>
      <label className="mb-1 block text-sm text-[var(--text-muted)]">Horizon: {horizon}</label>
      <Slider value={[horizon]} onValueChange={([v]) => setHorizon(v)} min={200} max={3000} step={100} className="mb-4" />
      <CanvasChart
        data={[
          { x: t, y: ucb, type: 'scatter', mode: 'lines', name: 'UCB1 (stochastic)', line: { color: '#4ade80', width: 3 } },
          { x: t, y: exp3, type: 'scatter', mode: 'lines', name: 'EXP3 (adversarial)', line: { color: '#60a5fa', width: 2 } },
          { x: t, y: epsGreedy, type: 'scatter', mode: 'lines', name: 'epsilon-greedy', line: { color: '#facc15', width: 2 } },
          { x: t, y: random, type: 'scatter', mode: 'lines', name: 'random', line: { color: '#f87171', width: 2, dash: 'dot' } },
        ]}
        layout={{
          title: { text: 'Typical cumulative regret scaling (illustrative)' },
          xaxis: { title: { text: 'round t' } },
          yaxis: { title: { text: 'cumulative regret' } },
          height: 420,
        }}
        style={{ width: '100%' }}
      />
    </div>
  );
}
