"use client";

import React, { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';
import { linspace, gaussianNoise } from '@/lib/math';

export default function LossLandscapeDemo({}: SimulationComponentProps): React.ReactElement {
  const [noise, setNoise] = useState(0.2);
  const [ripple, setRipple] = useState(2.5);
  const grid = useMemo(() => {
    const x = linspace(-3, 3, 45);
    const y = linspace(-3, 3, 45);
    const z = y.map((yy) =>
      x.map((xx, j) => {
        const seed = (yy + 7.1) * 1000 + j * 17.3 + ripple * 31;
        return xx * xx + 0.8 * yy * yy + 0.5 * Math.sin(ripple * xx) + gaussianNoise(seed, noise);
      })
    );
    return { x, y, z };
  }, [noise, ripple]);

  return (
    <SimulationPanel title="Loss Landscape">
      <SimulationConfig>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div>
            <SimulationLabel>Noise: {noise.toFixed(2)}</SimulationLabel>
            <Slider
              min={0}
              max={0.8}
              step={0.05}
              value={[noise]}
              onValueChange={([v]) => setNoise(v)}
            />
          </div>
          <div>
            <SimulationLabel>Ripple frequency: {ripple.toFixed(1)}</SimulationLabel>
            <Slider
              min={0.5}
              max={5}
              step={0.1}
              value={[ripple]}
              onValueChange={([v]) => setRipple(v)}
            />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
        <CanvasHeatmap
          data={[
            {
              z: grid.z,
              colorscale: 'Viridis',
              showscale: false,
            },
          ]}
          layout={{
            xaxis: { title: { text: 'w1' } },
            yaxis: { title: { text: 'w2' } },
            margin: { t: 20, r: 20, b: 45, l: 55 },
          }}
          style={{ width: '100%', height: 420 }}
        />
      </SimulationMain>
    </SimulationPanel>
  );
}
