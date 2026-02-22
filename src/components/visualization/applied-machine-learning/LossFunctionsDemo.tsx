"use client";

import React, { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';
import { linspace } from './ml-utils';

export default function LossFunctionsDemo({}: SimulationComponentProps): React.ReactElement {
  const [margin, setMargin] = useState(1);
  const x = useMemo(() => linspace(-3, 3, 200), []);
  const { zeroOne, hinge, bce, mse } = useMemo(() => {
    const z = x.map((v) => (v < 0 ? 1 : 0));
    const h = x.map((v) => Math.max(0, margin - v));
    const p = x.map((v) => 1 / (1 + Math.exp(-v)));
    const b = p.map((pv) => -Math.log(Math.max(pv, 1e-6)));
    const m = x.map((v) => v * v);
    return { zeroOne: z, hinge: h, bce: b, mse: m };
  }, [x, margin]);

  return (
    <SimulationPanel title="Loss Functions">
      <SimulationConfig>
        <div>
          <SimulationLabel>Hinge margin: {margin.toFixed(2)}</SimulationLabel>
          <Slider
            min={0.2}
            max={2}
            step={0.05}
            value={[margin]}
            onValueChange={([v]) => setMargin(v)}
          />
        </div>
      </SimulationConfig>
      <SimulationMain>
        <CanvasChart
          data={[
            { x, y: zeroOne, type: 'scatter', mode: 'lines', name: 'Zero-One' },
            { x, y: hinge, type: 'scatter', mode: 'lines', name: 'Hinge' },
            { x, y: bce, type: 'scatter', mode: 'lines', name: 'BCE (y=1)' },
            { x, y: mse, type: 'scatter', mode: 'lines', name: 'MSE' },
          ]}
          layout={{
            xaxis: { title: { text: 'Score/Residual' } },
            yaxis: { title: { text: 'Loss' } },
            margin: { t: 20, r: 20, b: 45, l: 55 },
          }}
          style={{ width: '100%', height: 360 }}
        />
      </SimulationMain>
    </SimulationPanel>
  );
}
