"use client";

import React, { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

function ringAdjacency(n: number, radius: number): number[][] {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      if (i === j) return 1;
      const d = Math.min((i - j + n) % n, (j - i + n) % n);
      return d <= radius ? 1 : 0;
    })
  );
}

export default function GraphMessagePassingDemo({}: SimulationComponentProps): React.ReactElement {
  const [n, setN] = useState(12);
  const [steps, setSteps] = useState(2);
  const [source, setSource] = useState(0);
  const series = useMemo(() => {
    const adj = ringAdjacency(n, 1);
    let signal: number[] = Array.from({ length: n }, (_, i) => (i === source ? 1 : 0));
    for (let s = 0; s < steps; s++) {
      const next = Array.from({ length: n }, () => 0);
      for (let i = 0; i < n; i++) {
        let acc = 0;
        let deg = 0;
        for (let j = 0; j < n; j++) {
          if (adj[i][j] > 0) {
            acc += signal[j];
            deg += 1;
          }
        }
        next[i] = deg > 0 ? acc / deg : signal[i];
      }
      signal = next;
    }
    return signal;
  }, [n, steps, source]);

  return (
    <SimulationPanel title="Message Passing on a Ring Graph">
      <SimulationConfig>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <div>
            <SimulationLabel>Nodes: {n}</SimulationLabel>
            <Slider min={6} max={24} step={1} value={[n]} onValueChange={([v]) => setN(v)} />
          </div>
          <div>
            <SimulationLabel>Steps: {steps}</SimulationLabel>
            <Slider min={1} max={8} step={1} value={[steps]} onValueChange={([v]) => setSteps(v)} />
          </div>
          <div>
            <SimulationLabel>Source node: {source}</SimulationLabel>
            <Slider min={0} max={Math.max(0, n - 1)} step={1} value={[source]} onValueChange={([v]) => setSource(v)} />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
        <CanvasChart
          data={[{ x: Array.from({ length: n }, (_, i) => i), y: series, type: 'bar', marker: { color: '#3b82f6' } }]}
          layout={{
            xaxis: { title: { text: 'Node index' } },
            yaxis: { title: { text: 'Signal after aggregation' } },
            margin: { t: 20, r: 20, b: 45, l: 55 },
          }}
          style={{ width: '100%', height: 360 }}
        />
      </SimulationMain>
    </SimulationPanel>
  );
}
