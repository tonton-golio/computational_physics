'use client';

import React, { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps { id: string }

export default function BanditRegretComparison({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [horizon, setHorizon] = useState(1000);
  const t = useMemo(() => Array.from({ length: horizon }, (_, i) => i + 1), [horizon]);
  const ucb = useMemo(() => t.map((x) => 0.8 * Math.sqrt(x * Math.log(x + 1))), [t]);
  const exp3 = useMemo(() => t.map((x) => 1.2 * Math.sqrt(4 * x * Math.log(4))), [t]);
  const epsGreedy = useMemo(() => t.map((x) => 0.06 * x + 8 * Math.sqrt(x)), [t]);
  const random = useMemo(() => t.map((x) => 0.22 * x), [t]);

  return (
    <div className="w-full rounded-lg bg-[#151525] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-3 text-white">Bandit Regret Comparison</h3>
      <label className="text-xs text-gray-300">Horizon: {horizon}<input className="w-full mb-4" type="range" min={200} max={3000} step={100} value={horizon} onChange={(e) => setHorizon(parseInt(e.target.value))} /></label>
      <Plot
        data={[
          { x: t, y: ucb, type: 'scatter', mode: 'lines', name: 'UCB1 (stochastic)', line: { color: '#4ade80', width: 3 } },
          { x: t, y: exp3, type: 'scatter', mode: 'lines', name: 'EXP3 (adversarial)', line: { color: '#60a5fa', width: 2 } },
          { x: t, y: epsGreedy, type: 'scatter', mode: 'lines', name: 'epsilon-greedy', line: { color: '#facc15', width: 2 } },
          { x: t, y: random, type: 'scatter', mode: 'lines', name: 'random', line: { color: '#f87171', width: 2, dash: 'dot' } },
        ]}
        layout={{
          title: { text: 'Typical cumulative regret scaling (illustrative)' },
          xaxis: { title: { text: 'round t' }, color: '#9ca3af' },
          yaxis: { title: { text: 'cumulative regret' }, color: '#9ca3af' },
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
