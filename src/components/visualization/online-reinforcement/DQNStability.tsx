'use client';

import React, { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps { id: string }

export default function DQNStability({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [steps, setSteps] = useState(300);
  const x = useMemo(() => Array.from({ length: steps }, (_, i) => i + 1), [steps]);
  const withReplay = useMemo(() => x.map((k) => 15 + 185 * (1 - Math.exp(-k / 90)) + 7 * Math.sin(k / 25)), [x]);
  const noReplay = useMemo(() => x.map((k) => 10 + 120 * (1 - Math.exp(-k / 120)) + 25 * Math.sin(k / 12)), [x]);
  const lossReplay = useMemo(() => x.map((k) => 0.8 * Math.exp(-k / 60) + 0.02 * Math.sin(k / 8)), [x]);
  const lossNoReplay = useMemo(() => x.map((k) => 0.95 * Math.exp(-k / 35) + 0.2 * Math.abs(Math.sin(k / 7))), [x]);

  return (
    <div className="w-full rounded-lg bg-[#151525] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-3 text-white">DQN Stability: Replay and Target Networks</h3>
      <label className="text-xs text-gray-300">Training points: {steps}<input className="w-full mb-4" type="range" min={100} max={1000} step={20} value={steps} onChange={(e) => setSteps(parseInt(e.target.value))} /></label>
      <Plot
        data={[
          { x, y: withReplay, type: 'scatter', mode: 'lines', name: 'Reward with replay/target net', line: { color: '#4ade80', width: 2 } },
          { x, y: noReplay, type: 'scatter', mode: 'lines', name: 'Reward without stabilization', line: { color: '#f87171', width: 2 } },
          { x, y: lossReplay.map((v) => v * 120), type: 'scatter', mode: 'lines', name: 'Loss with replay (scaled)', line: { color: '#60a5fa', width: 2, dash: 'dot' } },
          { x, y: lossNoReplay.map((v) => v * 120), type: 'scatter', mode: 'lines', name: 'Loss without replay (scaled)', line: { color: '#facc15', width: 2, dash: 'dot' } },
        ]}
        layout={{
          title: { text: 'Stabilization techniques reduce variance and divergence risk' },
          xaxis: { title: { text: 'training iteration' }, color: '#9ca3af' },
          yaxis: { title: { text: 'reward / scaled loss' }, color: '#9ca3af' },
          height: 430,
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
