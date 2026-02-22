"use client";

import { useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export default function FTLInstability({}: SimulationComponentProps) {
  const rounds = 200;
  const t = useMemo(() => Array.from({ length: rounds }, (_, i) => i + 1), []);
  const actions = useMemo(() => t.map((x) => (x % 2 === 0 ? 1 : 0)), [t]);
  const ftlLoss = useMemo(() => t.map((x) => x * 0.5 + (x % 2 === 0 ? 6 : 1)), [t]);
  const regularizedLoss = useMemo(() => t.map((x) => 0.26 * x + 4 * Math.sqrt(x)), [t]);

  return (
    <SimulationPanel title="FTL Instability Example" caption="In alternating adversarial losses, FTL can chase the last winner and oscillate.">
      <SimulationMain>
      <CanvasChart
        data={[
          { x: t, y: actions.map((a) => a * 4), type: 'scatter', mode: 'lines', name: 'FTL action (scaled)', line: { color: '#c084fc', width: 1 } },
          { x: t, y: ftlLoss, type: 'scatter', mode: 'lines', name: 'FTL cumulative loss', line: { color: '#f87171', width: 2 } },
          { x: t, y: regularizedLoss, type: 'scatter', mode: 'lines', name: 'Regularized baseline', line: { color: '#4ade80', width: 2 } },
        ]}
        layout={{
          title: { text: 'Oscillation under adversarial alternation' },
          xaxis: { title: { text: 'round t' } },
          yaxis: { title: { text: 'cumulative loss' } },
          height: 420,
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>
    </SimulationPanel>
  );
}
