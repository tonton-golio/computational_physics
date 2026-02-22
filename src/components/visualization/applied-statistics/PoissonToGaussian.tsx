"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

function poissonPMF(k: number, lam: number): number {
  // Use log to avoid overflow for large lambda
  let logP = k * Math.log(lam) - lam;
  for (let i = 2; i <= k; i++) logP -= Math.log(i);
  return Math.exp(logP);
}

export default function PoissonToGaussian({}: SimulationComponentProps) {
  const [lambda, setLambda] = useState(5);

  const { kVals, poissonVals, gaussVals } = useMemo(() => {
    const maxK = Math.max(20, Math.ceil(lambda + 4 * Math.sqrt(lambda)));
    const ks: number[] = [];
    const pVals: number[] = [];
    const gVals: number[] = [];
    const sig = Math.sqrt(lambda);
    for (let k = 0; k <= maxK; k++) {
      ks.push(k);
      pVals.push(poissonPMF(k, lambda));
      // Gaussian approximation
      const z = (k - lambda) / sig;
      gVals.push((1 / (sig * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * z * z));
    }
    return { kVals: ks, poissonVals: pVals, gaussVals: gVals };
  }, [lambda]);

  return (
    <SimulationPanel title="Poisson to Gaussian Convergence">
      <SimulationConfig>
        <div>
          <SimulationLabel>lambda: {lambda}</SimulationLabel>
          <Slider value={[lambda]} onValueChange={([v]) => setLambda(v)} min={1} max={100} step={1} />
        </div>
      </SimulationConfig>
      <SimulationMain>
      <CanvasChart
        data={[
          {
            x: kVals, y: poissonVals, type: 'bar',
            marker: { color: '#3b82f6' }, opacity: 0.6, name: 'Poisson',
          },
          {
            x: kVals, y: gaussVals, type: 'scatter', mode: 'lines',
            line: { color: '#ef4444', width: 2.5 }, name: 'Gaussian approx.',
          },
        ]}
        layout={{
          height: 400,
          xaxis: { title: { text: 'k' } },
          yaxis: { title: { text: 'P(k)' } },
          barmode: 'overlay',
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>
    </SimulationPanel>
  );
}
