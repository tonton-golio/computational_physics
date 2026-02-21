'use client';

import React, { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import type { SimulationComponentProps } from '@/shared/types/simulation';


function smooth(values: number[], window: number): number[] {
  return values.map((_, i) => {
    const start = Math.max(0, i - window + 1);
    const slice = values.slice(start, i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

export default function CartPoleLearningCurves({ id }: SimulationComponentProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
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
    <div className="w-full rounded-lg bg-[var(--surface-1)] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-3 text-[var(--text-strong)]">CartPole Learning Progress</h3>
      <p className="text-sm text-[var(--text-muted)] mb-4">
        Episode duration improves as the policy learns to keep the pole balanced for longer horizons.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Episodes: {episodes}</label>
          <Slider value={[episodes]} onValueChange={([v]) => setEpisodes(v)} min={100} max={1000} step={25} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Noise level: {noise}</label>
          <Slider value={[noise]} onValueChange={([v]) => setNoise(v)} min={0} max={60} step={1} />
        </div>
      </div>
      <CanvasChart
        data={[
          { x: Array.from({ length: raw.length }, (_, i) => i + 1), y: raw, type: 'scatter', mode: 'lines', name: 'Episode duration', line: { color: '#64748b', width: 1 } },
          { x: Array.from({ length: avg.length }, (_, i) => i + 1), y: avg, type: 'scatter', mode: 'lines', name: '20-episode moving average', line: { color: '#4ade80', width: 3 } },
        ]}
        layout={{
          title: { text: 'Training curve (synthetic CartPole-like progression)' },
          xaxis: { title: { text: 'episode' } },
          yaxis: { title: { text: 'duration (steps)' }, range: [0, 520] },
          height: 420,
          margin: { t: 40, b: 60, l: 60, r: 20 },
        }}
        style={{ width: '100%' }}
      />
    </div>
  );
}
