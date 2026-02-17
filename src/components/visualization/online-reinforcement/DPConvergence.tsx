'use client';

import React, { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps { id: string }

export default function DPConvergence({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [iters, setIters] = useState(40);
  const t = useMemo(() => Array.from({ length: iters }, (_, i) => i + 1), [iters]);
  const valueResidual = useMemo(() => t.map((k) => Math.exp(-k / 6)), [t]);
  const policyChanges = useMemo(() => t.map((k) => Math.max(0, 30 - k + (k % 6 === 0 ? 2 : 0))), [t]);

  return (
    <div className="w-full rounded-lg bg-[#151525] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-3 text-white">Value Iteration / Policy Iteration Convergence</h3>
      <label className="text-xs text-gray-300">Iterations: {iters}<input className="w-full mb-4" type="range" min={10} max={150} step={5} value={iters} onChange={(e) => setIters(parseInt(e.target.value))} /></label>
      <Plot
        data={[
          { x: t, y: valueResidual, type: 'scatter', mode: 'lines+markers', name: 'Value iteration Bellman residual', line: { color: '#4ade80', width: 2 } },
          { x: t, y: policyChanges.map((v) => Math.log10(v + 1)), type: 'scatter', mode: 'lines+markers', name: 'log10(policy changes + 1)', line: { color: '#60a5fa', width: 2 } },
        ]}
        layout={{
          title: { text: 'Contraction and policy stabilization (illustrative)' },
          xaxis: { title: { text: 'iteration' }, color: '#9ca3af' },
          yaxis: { title: { text: 'residual / log policy updates' }, type: 'log', color: '#9ca3af' },
          height: 420,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(15,15,25,1)',
          font: { color: '#9ca3af' },
        }}
        config={{ displayModeBar: false }}
        style={{ width: '100%' }}
      />
    </div>
  );
}
