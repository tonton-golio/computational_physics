"use client";

import React, { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

export default function ExplainedVarianceDemo({}: SimulationComponentProps): React.ReactElement {
  const [dims, setDims] = useState(8);
  const eig = useMemo(() => {
    const values = Array.from({ length: dims }, (_, i) => Math.exp(-i / 2.1));
    const total = values.reduce((a, b) => a + b, 0);
    let run = 0;
    const cumulative = values.map((v) => {
      run += v;
      return run / total;
    });
    return { values: values.map((v) => v / total), cumulative };
  }, [dims]);

  const xAxis = Array.from({ length: dims }, (_, i) => i + 1);

  return (
    <SimulationPanel title="Explained Variance (Scree)">
      <SimulationConfig>
        <div>
          <SimulationLabel>Dimensions: {dims}</SimulationLabel>
          <Slider min={3} max={20} step={1} value={[dims]} onValueChange={([v]) => setDims(v)} />
        </div>
      </SimulationConfig>
      <SimulationMain>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div>
            <p className="mb-1 text-center text-sm text-[var(--text-muted)]">Per-component variance</p>
            <CanvasChart
              data={[
                { x: xAxis, y: eig.values, type: 'bar', name: 'Per-component' },
              ]}
              layout={{
                xaxis: { title: { text: 'Component' } },
                yaxis: { title: { text: 'Explained variance ratio' } },
                margin: { t: 20, r: 20, b: 45, l: 55 },
              }}
              style={{ width: '100%', height: 360 }}
            />
          </div>
          <div>
            <p className="mb-1 text-center text-sm text-[var(--text-muted)]">Cumulative variance</p>
            <CanvasChart
              data={[
                { x: xAxis, y: eig.cumulative, type: 'scatter', mode: 'lines+markers', name: 'Cumulative' },
              ]}
              layout={{
                xaxis: { title: { text: 'Component' } },
                yaxis: { title: { text: 'Cumulative variance' }, range: [0, 1.05] },
                margin: { t: 20, r: 20, b: 45, l: 55 },
              }}
              style={{ width: '100%', height: 360 }}
            />
          </div>
        </div>
      </SimulationMain>
    </SimulationPanel>
  );
}
