"use client";

import { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationLabel, SimulationButton, SimulationSettings, SimulationConfig } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';


function mulberry32(a: number) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export default function MonteCarloConvergence({}: SimulationComponentProps) {
  const [episodes, setEpisodes] = useState(500);
  const [seed, setSeed] = useState(11);
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
    <SimulationPanel title="Monte Carlo Value Estimation Convergence">
      <SimulationSettings>
        <SimulationButton variant="primary" onClick={() => setSeed((s) => s + 1)}>Re-sample trajectories</SimulationButton>
      </SimulationSettings>
      <SimulationConfig>
        <div>
          <SimulationLabel>Episodes: {episodes}</SimulationLabel>
          <Slider value={[episodes]} onValueChange={([v]) => setEpisodes(v)} min={50} max={2000} step={50} />
        </div>
      </SimulationConfig>
      <SimulationMain>
      <CanvasChart
        data={[
          { x, y: estimates, type: 'scatter', mode: 'lines', name: 'MC estimate', line: { color: '#60a5fa', width: 2 } },
          { x, y: truth, type: 'scatter', mode: 'lines', name: 'True value', line: { color: '#4ade80', width: 2, dash: 'dot' } },
        ]}
        layout={{
          title: { text: 'Law of large numbers in episodic returns' },
          xaxis: { title: { text: 'episode' } },
          yaxis: { title: { text: 'V(s) estimate' } },
          height: 420,
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>
    </SimulationPanel>
  );
}
