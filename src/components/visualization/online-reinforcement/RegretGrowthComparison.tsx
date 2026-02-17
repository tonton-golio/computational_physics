'use client';

import React, { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps { id: string }

export default function RegretGrowthComparison({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [horizon, setHorizon] = useState(1000);

  const t = useMemo(() => Array.from({ length: horizon }, (_, i) => i + 1), [horizon]);
  const sublinear = useMemo(() => t.map((x) => 2.5 * Math.sqrt(x)), [t]);
  const linear = useMemo(() => t.map((x) => 0.35 * x), [t]);
  const bestExpertGap = useMemo(() => t.map((x) => 0.18 * x + 9 * Math.sin(x / 80)), [t]);

  return (
    <div className="w-full rounded-lg bg-[#151525] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-3 text-white">Regret Growth Over Time</h3>
      <label className="text-xs text-gray-300">Horizon: {horizon}<input className="w-full mb-4" type="range" min={200} max={3000} step={100} value={horizon} onChange={(e) => setHorizon(parseInt(e.target.value))} /></label>
      <Plot
        data={[
          { x: t, y: sublinear, type: 'scatter', mode: 'lines', name: 'Sublinear regret O(sqrt(T))', line: { color: '#4ade80', width: 3 } },
          { x: t, y: linear, type: 'scatter', mode: 'lines', name: 'Linear regret O(T)', line: { color: '#f87171', width: 2 } },
          { x: t, y: bestExpertGap, type: 'scatter', mode: 'lines', name: 'Learner vs best expert (example)', line: { color: '#60a5fa', width: 2, dash: 'dot' } },
        ]}
        layout={{
          title: { text: 'No-regret behavior requires sublinear growth' },
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
