"use client";

import React, { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';
import { pseudoRandom } from './ml-utils';

export default function GraphAdjacencyDemo({}: SimulationComponentProps): React.ReactElement {
  const [n, setN] = useState(10);
  const [p, setP] = useState(0.25);
  const mat = useMemo(() => {
    const m = Array.from({ length: n }, () => Array.from({ length: n }, () => 0));
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        if (i === j) {
          m[i][j] = 1;
        } else if (pseudoRandom(i * 97 + j * 57 + n * 11 + p * 1000) < p) {
          m[i][j] = 1;
          m[j][i] = 1;
        }
      }
    }
    return m;
  }, [n, p]);

  return (
    <SimulationPanel title="Random Graph Adjacency">
      <SimulationConfig>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div>
            <SimulationLabel>Nodes: {n}</SimulationLabel>
            <Slider min={6} max={25} step={1} value={[n]} onValueChange={([v]) => setN(v)} />
          </div>
          <div>
            <SimulationLabel>Edge probability: {p.toFixed(2)}</SimulationLabel>
            <Slider min={0.05} max={0.7} step={0.01} value={[p]} onValueChange={([v]) => setP(v)} />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
        <CanvasHeatmap
          data={[{ z: mat, colorscale: 'Viridis', showscale: false }]}
          layout={{
            xaxis: { title: { text: 'Node' } },
            yaxis: { title: { text: 'Node' } },
            margin: { t: 20, r: 20, b: 45, l: 55 },
          }}
          style={{ width: '100%', height: 360 }}
        />
      </SimulationMain>
    </SimulationPanel>
  );
}
