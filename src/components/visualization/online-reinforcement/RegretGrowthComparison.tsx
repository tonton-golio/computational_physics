'use client';

import React, { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export default function RegretGrowthComparison({ id }: SimulationComponentProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [horizon, setHorizon] = useState(1000);

  const t = useMemo(() => Array.from({ length: horizon }, (_, i) => i + 1), [horizon]);
  const sublinear = useMemo(() => t.map((x) => 2.5 * Math.sqrt(x)), [t]);
  const linear = useMemo(() => t.map((x) => 0.35 * x), [t]);
  const bestExpertGap = useMemo(() => t.map((x) => 0.18 * x + 9 * Math.sin(x / 80)), [t]);

  return (
    <div className="w-full rounded-lg bg-[var(--surface-1)] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-3 text-[var(--text-strong)]">Regret Growth Over Time</h3>
      <label className="mb-1 block text-sm text-[var(--text-muted)]">Horizon: {horizon}</label>
      <Slider value={[horizon]} onValueChange={([v]) => setHorizon(v)} min={200} max={3000} step={100} className="mb-4" />
      <CanvasChart
        data={[
          { x: t, y: sublinear, type: 'scatter', mode: 'lines', name: 'Sublinear regret O(sqrt(T))', line: { color: '#4ade80', width: 3 } },
          { x: t, y: linear, type: 'scatter', mode: 'lines', name: 'Linear regret O(T)', line: { color: '#f87171', width: 2 } },
          { x: t, y: bestExpertGap, type: 'scatter', mode: 'lines', name: 'Learner vs best expert (example)', line: { color: '#60a5fa', width: 2, dash: 'dot' } },
        ]}
        layout={{
          title: { text: 'No-regret behavior requires sublinear growth' },
          xaxis: { title: { text: 'round t' } },
          yaxis: { title: { text: 'cumulative regret' } },
          height: 420,
        }}
        style={{ width: '100%' }}
      />
    </div>
  );
}
