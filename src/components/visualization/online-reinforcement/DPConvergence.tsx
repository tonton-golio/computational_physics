'use client';

import React, { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export default function DPConvergence({ id }: SimulationComponentProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [iters, setIters] = useState(40);
  const t = useMemo(() => Array.from({ length: iters }, (_, i) => i + 1), [iters]);
  const valueResidual = useMemo(() => t.map((k) => Math.exp(-k / 6)), [t]);
  const policyChanges = useMemo(() => t.map((k) => Math.max(0, 30 - k + (k % 6 === 0 ? 2 : 0))), [t]);

  return (
    <div className="w-full rounded-lg bg-[var(--surface-1)] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-3 text-[var(--text-strong)]">Value Iteration / Policy Iteration Convergence</h3>
      <label className="mb-1 block text-sm text-[var(--text-muted)]">Iterations: {iters}</label>
      <Slider value={[iters]} onValueChange={([v]) => setIters(v)} min={10} max={150} step={5} className="mb-4" />
      <CanvasChart
        data={[
          { x: t, y: valueResidual, type: 'scatter', mode: 'lines+markers', name: 'Value iteration Bellman residual', line: { color: '#4ade80', width: 2 } },
          { x: t, y: policyChanges.map((v) => Math.log10(v + 1)), type: 'scatter', mode: 'lines+markers', name: 'log10(policy changes + 1)', line: { color: '#60a5fa', width: 2 } },
        ]}
        layout={{
          title: { text: 'Contraction and policy stabilization (illustrative)' },
          xaxis: { title: { text: 'iteration' } },
          yaxis: { title: { text: 'residual / log policy updates' }, type: 'log' },
          height: 420,
        }}
        style={{ width: '100%' }}
      />
    </div>
  );
}
