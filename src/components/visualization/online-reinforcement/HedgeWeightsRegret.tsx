"use client";

import { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationLabel, SimulationConfig } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export default function HedgeWeightsRegret({}: SimulationComponentProps) {
  const [eta, setEta] = useState(0.3);
  const rounds = 220;
  const experts = 4;

  const { probs, regret } = useMemo(() => {
    const w = Array.from({ length: experts }, () => 1);
    const history = Array.from({ length: experts }, () => [] as number[]);
    let learnerLoss = 0;
    const expertLoss = new Array(experts).fill(0);
    const reg: number[] = [];

    for (let t = 1; t <= rounds; t++) {
      const z = w.reduce((a, b) => a + b, 0);
      const p = w.map((wi) => wi / z);
      const losses = [0.45 + 0.1 * Math.sin(t / 17), 0.52, 0.6 - 0.12 * Math.cos(t / 19), 0.5 + (t % 35 === 0 ? 0.35 : 0)];
      const pLoss = p.reduce((sum, pi, i) => sum + pi * losses[i], 0);
      learnerLoss += pLoss;
      for (let i = 0; i < experts; i++) {
        expertLoss[i] += losses[i];
        w[i] = w[i] * Math.exp(-eta * losses[i]);
      }
      const zNew = w.reduce((a, b) => a + b, 0);
      for (let i = 0; i < experts; i++) history[i].push(w[i] / zNew);
      reg.push(learnerLoss - Math.min(...expertLoss));
    }
    return { probs: history, regret: reg };
  }, [eta]);

  const x = Array.from({ length: rounds }, (_, i) => i + 1);

  return (
    <SimulationPanel title="Hedge: Weight Evolution and Regret">
      <SimulationConfig>
        <div>
          <SimulationLabel>Learning rate eta: {eta.toFixed(2)}</SimulationLabel>
          <Slider value={[eta]} onValueChange={([v]) => setEta(v)} min={0.05} max={1} step={0.05} />
        </div>
      </SimulationConfig>
      <SimulationMain>
      <CanvasChart
        data={[
          ...probs.map((series, i) => ({ x, y: series.map((v) => v * 10), type: 'scatter' as const, mode: 'lines' as const, name: `Expert ${i + 1} prob (scaled)`, })),
          { x, y: regret, type: 'scatter' as const, mode: 'lines' as const, name: 'Regret to best expert', line: { color: '#4ade80', width: 3 } },
        ]}
        layout={{
          title: { text: 'Multiplicative updates concentrate on stronger experts' },
          xaxis: { title: { text: 'round t' } },
          yaxis: { title: { text: 'regret / scaled probabilities' } },
          height: 430,
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>
    </SimulationPanel>
  );
}
