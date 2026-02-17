'use client';

import React, { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps { id: string }

export default function AverageRewardVsDiscounted({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [horizon, setHorizon] = useState(800);
  const x = useMemo(() => Array.from({ length: horizon }, (_, i) => i + 1), [horizon]);
  const discounted = useMemo(() => x.map((t) => 45 + 40 * (1 - Math.exp(-t / 180)) + 6 * Math.sin(t / 60)), [x]);
  const averageReward = useMemo(() => x.map((t) => 20 + 0.06 * t + 2 * Math.sin(t / 45)), [x]);

  return (
    <div className="w-full rounded-lg bg-[#151525] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-3 text-white">Average-Reward vs Discounted Learning</h3>
      <label className="text-xs text-gray-300">Horizon: {horizon}<input className="w-full mb-4" type="range" min={200} max={3000} step={100} value={horizon} onChange={(e) => setHorizon(parseInt(e.target.value))} /></label>
      <Plot
        data={[
          { x, y: discounted, type: 'scatter', mode: 'lines', name: 'Discounted objective performance', line: { color: '#60a5fa', width: 2 } },
          { x, y: averageReward, type: 'scatter', mode: 'lines', name: 'Average reward gain estimate', line: { color: '#4ade80', width: 2 } },
        ]}
        layout={{
          title: { text: 'Continuing-task metrics emphasize different objectives' },
          xaxis: { title: { text: 'time step' }, color: '#9ca3af' },
          yaxis: { title: { text: 'performance metric' }, color: '#9ca3af' },
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
