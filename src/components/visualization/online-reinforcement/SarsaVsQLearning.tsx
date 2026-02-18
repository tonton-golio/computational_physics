'use client';

import React, { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';
import { Slider } from '@/components/ui/slider';
import { usePlotlyTheme } from '@/lib/plotly-theme';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps { id: string }

export default function SarsaVsQLearning({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [episodes, setEpisodes] = useState(250);
  const { mergeLayout } = usePlotlyTheme();
  const x = useMemo(() => Array.from({ length: episodes }, (_, i) => i + 1), [episodes]);
  const sarsa = useMemo(() => x.map((ep) => -120 + 90 * (1 - Math.exp(-ep / 90)) - 6 * Math.sin(ep / 18)), [x]);
  const qLearning = useMemo(() => x.map((ep) => -150 + 130 * (1 - Math.exp(-ep / 70)) + 4 * Math.sin(ep / 15)), [x]);

  return (
    <div className="w-full rounded-lg bg-[var(--surface-1)] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-3 text-[var(--text-strong)]">SARSA vs Q-Learning (Cliff-like Task)</h3>
      <label className="mb-1 block text-sm text-[var(--text-muted)]">Episodes: {episodes}</label>
      <Slider value={[episodes]} onValueChange={([v]) => setEpisodes(v)} min={50} max={1000} step={25} className="mb-4" />
      <Plot
        data={[
          { x, y: sarsa, type: 'scatter', mode: 'lines', name: 'SARSA (on-policy)', line: { color: '#facc15', width: 2 } },
          { x, y: qLearning, type: 'scatter', mode: 'lines', name: 'Q-learning (off-policy)', line: { color: '#4ade80', width: 2 } },
        ]}
        layout={mergeLayout({
          title: { text: 'Typical return trajectories (illustrative)' },
          xaxis: { title: { text: 'episode' } },
          yaxis: { title: { text: 'return per episode' } },
          height: 420,
        })}
        config={{ displayModeBar: false }}
        style={{ width: '100%' }}
      />
    </div>
  );
}
