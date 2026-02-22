"use client";

import { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationLabel, SimulationConfig } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export default function SarsaVsQLearning({}: SimulationComponentProps) {
  const [episodes, setEpisodes] = useState(250);
  const x = useMemo(() => Array.from({ length: episodes }, (_, i) => i + 1), [episodes]);
  const sarsa = useMemo(() => x.map((ep) => -120 + 90 * (1 - Math.exp(-ep / 90)) - 6 * Math.sin(ep / 18)), [x]);
  const qLearning = useMemo(() => x.map((ep) => -150 + 130 * (1 - Math.exp(-ep / 70)) + 4 * Math.sin(ep / 15)), [x]);

  return (
    <SimulationPanel title="SARSA vs Q-Learning (Cliff-like Task)">
      <SimulationConfig>
        <div>
          <SimulationLabel>Episodes: {episodes}</SimulationLabel>
          <Slider value={[episodes]} onValueChange={([v]) => setEpisodes(v)} min={50} max={1000} step={25} />
        </div>
      </SimulationConfig>
      <SimulationMain>
      <CanvasChart
        data={[
          { x, y: sarsa, type: 'scatter', mode: 'lines', name: 'SARSA (on-policy)', line: { color: '#facc15', width: 2 } },
          { x, y: qLearning, type: 'scatter', mode: 'lines', name: 'Q-learning (off-policy)', line: { color: '#4ade80', width: 2 } },
        ]}
        layout={{
          title: { text: 'Typical return trajectories (illustrative)' },
          xaxis: { title: { text: 'episode' } },
          yaxis: { title: { text: 'return per episode' } },
          height: 420,
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>
    </SimulationPanel>
  );
}
