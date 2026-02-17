'use client';

import React, { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id: string;
}

function smooth(values: number[], window: number): number[] {
  return values.map((_, i) => {
    const start = Math.max(0, i - window + 1);
    const slice = values.slice(start, i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

export default function CartPoleLearningCurves({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [episodes, setEpisodes] = useState(250);
  const [noise, setNoise] = useState(20);

  const { raw, avg } = useMemo(() => {
    const rawDurations = Array.from({ length: episodes }, (_, ep) => {
      const trend = 20 + 180 * (1 - Math.exp(-ep / 70));
      const wobble = Math.sin(ep / 7) * 8 + Math.sin(ep / 17) * 10;
      const pseudo = Math.sin(ep * 12.9898 + 78.233) * 43758.5453;
      const fractional = pseudo - Math.floor(pseudo);
      const jitter = (fractional - 0.5) * 2 * noise;
      return Math.max(5, Math.min(500, trend + wobble + jitter));
    });
    return { raw: rawDurations, avg: smooth(rawDurations, 20) };
  }, [episodes, noise]);

  return (
    <div className="w-full rounded-lg bg-[#151525] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-3 text-white">CartPole Learning Progress</h3>
      <p className="text-sm text-gray-400 mb-4">
        Episode duration improves as the policy learns to keep the pole balanced for longer horizons.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
        <label className="text-xs text-gray-300">Episodes: {episodes}<input className="w-full" type="range" min={100} max={1000} step={25} value={episodes} onChange={(e) => setEpisodes(parseInt(e.target.value))} /></label>
        <label className="text-xs text-gray-300">Noise level: {noise}<input className="w-full" type="range" min={0} max={60} step={1} value={noise} onChange={(e) => setNoise(parseInt(e.target.value))} /></label>
      </div>
      <Plot
        data={[
          { x: Array.from({ length: raw.length }, (_, i) => i + 1), y: raw, type: 'scatter', mode: 'lines', name: 'Episode duration', line: { color: '#64748b', width: 1 } },
          { x: Array.from({ length: avg.length }, (_, i) => i + 1), y: avg, type: 'scatter', mode: 'lines', name: '20-episode moving average', line: { color: '#4ade80', width: 3 } },
        ]}
        layout={{
          title: { text: 'Training curve (synthetic CartPole-like progression)' },
          xaxis: { title: { text: 'episode' }, color: '#9ca3af' },
          yaxis: { title: { text: 'duration (steps)' }, color: '#9ca3af', range: [0, 520] },
          height: 420,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(15,15,25,1)',
          font: { color: '#9ca3af' },
          margin: { t: 40, b: 60, l: 60, r: 20 },
        }}
        config={{ displayModeBar: false }}
        style={{ width: '100%' }}
      />
    </div>
  );
}
