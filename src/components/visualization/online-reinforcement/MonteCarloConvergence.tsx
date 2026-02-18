'use client';

import React, { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';
import { Slider } from '@/components/ui/slider';
import { usePlotlyTheme } from '@/lib/plotly-theme';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps { id: string }

function mulberry32(a: number) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export default function MonteCarloConvergence({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [episodes, setEpisodes] = useState(500);
  const [seed, setSeed] = useState(11);
  const { mergeLayout } = usePlotlyTheme();
  const trueValue = 1.0;

  const estimates = useMemo(() => {
    const rng = mulberry32(seed);
    let avg = 0;
    const out: number[] = [];
    for (let n = 1; n <= episodes; n++) {
      const sample = trueValue + (rng() - 0.5) * 1.6;
      avg += (sample - avg) / n;
      out.push(avg);
    }
    return out;
  }, [episodes, seed]);

  const x = Array.from({ length: episodes }, (_, i) => i + 1);
  const truth = x.map(() => trueValue);

  return (
    <div className="w-full rounded-lg bg-[var(--surface-1)] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-3 text-[var(--text-strong)]">Monte Carlo Value Estimation Convergence</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Episodes: {episodes}</label>
          <Slider value={[episodes]} onValueChange={([v]) => setEpisodes(v)} min={50} max={2000} step={50} />
        </div>
        <button onClick={() => setSeed((s) => s + 1)} className="rounded bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white text-sm px-3 py-2">Re-sample trajectories</button>
      </div>
      <Plot
        data={[
          { x, y: estimates, type: 'scatter', mode: 'lines', name: 'MC estimate', line: { color: '#60a5fa', width: 2 } },
          { x, y: truth, type: 'scatter', mode: 'lines', name: 'True value', line: { color: '#4ade80', width: 2, dash: 'dot' } },
        ]}
        layout={mergeLayout({
          title: { text: 'Law of large numbers in episodic returns' },
          xaxis: { title: { text: 'episode' } },
          yaxis: { title: { text: 'V(s) estimate' } },
          height: 420,
        })}
        config={{ displayModeBar: false }}
        style={{ width: '100%' }}
      />
    </div>
  );
}
