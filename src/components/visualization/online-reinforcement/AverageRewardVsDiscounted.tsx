"use client";

import { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationLabel, SimulationConfig } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export default function AverageRewardVsDiscounted({}: SimulationComponentProps) {
  const [horizon, setHorizon] = useState(800);
  const x = useMemo(() => Array.from({ length: horizon }, (_, i) => i + 1), [horizon]);
  const discounted = useMemo(() => x.map((t) => 45 + 40 * (1 - Math.exp(-t / 180)) + 6 * Math.sin(t / 60)), [x]);
  const averageReward = useMemo(() => x.map((t) => 20 + 0.06 * t + 2 * Math.sin(t / 45)), [x]);

  return (
    <SimulationPanel title="Average-Reward vs Discounted Learning">
      <SimulationConfig>
        <div>
          <SimulationLabel>Horizon: {horizon}</SimulationLabel>
          <Slider value={[horizon]} onValueChange={([v]) => setHorizon(v)} min={200} max={3000} step={100} />
        </div>
      </SimulationConfig>
      <SimulationMain>
      <CanvasChart
        data={[
          { x, y: discounted, type: 'scatter', mode: 'lines', name: 'Discounted objective performance', line: { color: '#60a5fa', width: 2 } },
          { x, y: averageReward, type: 'scatter', mode: 'lines', name: 'Average reward gain estimate', line: { color: '#4ade80', width: 2 } },
        ]}
        layout={{
          title: { text: 'Continuing-task metrics emphasize different objectives' },
          xaxis: { title: { text: 'time step' } },
          yaxis: { title: { text: 'performance metric' } },
          height: 420,
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>
    </SimulationPanel>
  );
}
