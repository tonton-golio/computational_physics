'use client';

import React, { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps { id: string }

export default function SarsaVsQLearning({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [episodes, setEpisodes] = useState(250);
  const x = useMemo(() => Array.from({ length: episodes }, (_, i) => i + 1), [episodes]);
  const sarsa = useMemo(() => x.map((ep) => -120 + 90 * (1 - Math.exp(-ep / 90)) - 6 * Math.sin(ep / 18)), [x]);
  const qLearning = useMemo(() => x.map((ep) => -150 + 130 * (1 - Math.exp(-ep / 70)) + 4 * Math.sin(ep / 15)), [x]);

  return (
    <div className="w-full rounded-lg bg-[#151525] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-3 text-white">SARSA vs Q-Learning (Cliff-like Task)</h3>
      <label className="text-xs text-gray-300">Episodes: {episodes}<input className="w-full mb-4" type="range" min={50} max={1000} step={25} value={episodes} onChange={(e) => setEpisodes(parseInt(e.target.value))} /></label>
      <Plot
        data={[
          { x, y: sarsa, type: 'scatter', mode: 'lines', name: 'SARSA (on-policy)', line: { color: '#facc15', width: 2 } },
          { x, y: qLearning, type: 'scatter', mode: 'lines', name: 'Q-learning (off-policy)', line: { color: '#4ade80', width: 2 } },
        ]}
        layout={{
          title: { text: 'Typical return trajectories (illustrative)' },
          xaxis: { title: { text: 'episode' }, color: '#9ca3af' },
          yaxis: { title: { text: 'return per episode' }, color: '#9ca3af' },
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
