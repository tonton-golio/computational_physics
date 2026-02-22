"use client";

import { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationLabel, SimulationConfig } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export default function DQNStability({}: SimulationComponentProps) {
  const [steps, setSteps] = useState(300);
  const x = useMemo(() => Array.from({ length: steps }, (_, i) => i + 1), [steps]);
  const withReplay = useMemo(() => x.map((k) => 15 + 185 * (1 - Math.exp(-k / 90)) + 7 * Math.sin(k / 25)), [x]);
  const noReplay = useMemo(() => x.map((k) => 10 + 120 * (1 - Math.exp(-k / 120)) + 25 * Math.sin(k / 12)), [x]);
  const lossReplay = useMemo(() => x.map((k) => 0.8 * Math.exp(-k / 60) + 0.02 * Math.sin(k / 8)), [x]);
  const lossNoReplay = useMemo(() => x.map((k) => 0.95 * Math.exp(-k / 35) + 0.2 * Math.abs(Math.sin(k / 7))), [x]);

  return (
    <SimulationPanel title="DQN Stability: Replay and Target Networks">
      <SimulationConfig>
        <div>
          <SimulationLabel>Training points: {steps}</SimulationLabel>
          <Slider value={[steps]} onValueChange={([v]) => setSteps(v)} min={100} max={1000} step={20} />
        </div>
      </SimulationConfig>
      <SimulationMain>
      <CanvasChart
        data={[
          { x, y: withReplay, type: 'scatter', mode: 'lines', name: 'Reward with replay/target net', line: { color: '#4ade80', width: 2 } },
          { x, y: noReplay, type: 'scatter', mode: 'lines', name: 'Reward without stabilization', line: { color: '#f87171', width: 2 } },
          { x, y: lossReplay.map((v) => v * 120), type: 'scatter', mode: 'lines', name: 'Loss with replay (scaled)', line: { color: '#60a5fa', width: 2, dash: 'dot' } },
          { x, y: lossNoReplay.map((v) => v * 120), type: 'scatter', mode: 'lines', name: 'Loss without replay (scaled)', line: { color: '#facc15', width: 2, dash: 'dot' } },
        ]}
        layout={{
          title: { text: 'Stabilization techniques reduce variance and divergence risk' },
          xaxis: { title: { text: 'training iteration' } },
          yaxis: { title: { text: 'reward / scaled loss' } },
          height: 430,
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>
    </SimulationPanel>
  );
}
