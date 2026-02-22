"use client";

import React, { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
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

export default function GraphConvolutionIntuitionDemo({}: SimulationComponentProps): React.ReactElement {
  const [n, setN] = useState(12);
  const [radius, setRadius] = useState(1);
  const z = useMemo(() => ringAdjacency(n, radius), [n, radius]);

  return (
    <SimulationPanel title="Normalized Adjacency Intuition">
      <SimulationConfig>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div>
            <SimulationLabel>Nodes: {n}</SimulationLabel>
            <Slider min={6} max={24} step={1} value={[n]} onValueChange={([v]) => setN(v)} />
          </div>
          <div>
            <SimulationLabel>Neighborhood radius: {radius}</SimulationLabel>
            <Slider min={1} max={4} step={1} value={[radius]} onValueChange={([v]) => setRadius(v)} />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
        <CanvasHeatmap
          data={[{ z, colorscale: 'Magma', showscale: false }]}
          layout={{
            xaxis: { title: { text: 'Node index' } },
            yaxis: { title: { text: 'Node index' } },
            margin: { t: 20, r: 20, b: 45, l: 55 },
          }}
          style={{ width: '100%', height: 360 }}
        />
      </SimulationMain>
    </SimulationPanel>
  );
}
