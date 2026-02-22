"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/** Log-space factorial */
function logFactorial(n: number): number {
  if (n <= 1) return 0;
  let s = 0;
  for (let i = 2; i <= n; i++) s += Math.log(i);
  return s;
}

/** Poisson PMF: P(k; lambda) = lambda^k * exp(-lambda) / k! */
function poissonPMF(k: number, lambda: number): number {
  if (lambda <= 0) return k === 0 ? 1 : 0;
  const logP = k * Math.log(lambda) - lambda - logFactorial(k);
  return Math.exp(logP);
}

export default function PoissonDistribution({}: SimulationComponentProps) {
  const [m1, setM1] = useState(1);
  const [m2, setM2] = useState(10);

  const { kVals, y1, y2 } = useMemo(() => {
    const kMax = 50;
    const kVals: number[] = [];
    const y1: number[] = [];
    const y2: number[] = [];

    for (let k = 0; k <= kMax; k++) {
      kVals.push(k);
      y1.push(poissonPMF(k, m1));
      y2.push(poissonPMF(k, m2));
    }

    return { kVals, y1, y2 };
  }, [m1, m2]);

  return (
    <SimulationPanel title="Poisson Distribution">
      <SimulationConfig>
        <div>
          <SimulationLabel>Mean m1: {m1}</SimulationLabel>
          <Slider value={[m1]} onValueChange={([v]) => setM1(v)} min={1} max={32} step={1} />
        </div>
        <div>
          <SimulationLabel>Mean m2: {m2}</SimulationLabel>
          <Slider value={[m2]} onValueChange={([v]) => setM2(v)} min={1} max={32} step={1} />
        </div>
      </SimulationConfig>

      <SimulationMain>
      <CanvasChart
        data={[
          {
            x: kVals, y: y1, type: 'scatter', mode: 'lines+markers',
            marker: { size: 5, color: '#f97316' },
            line: { color: '#f97316', width: 2 },
            name: `m=${m1}`,
          },
          {
            x: kVals, y: y2, type: 'scatter', mode: 'lines+markers',
            marker: { size: 5, color: '#3b82f6' },
            line: { color: '#3b82f6', width: 2 },
            name: `m=${m2}`,
          },
        ] as any}
        layout={{
          height: 400,
          margin: { t: 40, b: 50, l: 60, r: 20 },
          title: { text: 'Poisson Distribution P_m(k)' },
          xaxis: {
            title: { text: 'k' },
            range: [0, 50],
          },
          yaxis: {
            title: { text: 'P(k)' },
            range: [0, 0.4],
          },
          legend: { x: 0.65, y: 0.95, bgcolor: 'rgba(0,0,0,0.3)' },
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>

      <div className="mt-3 text-sm text-[var(--text-muted)]">
        <p>
          The <strong className="text-[var(--text-muted)]">Poisson distribution</strong> models the number of events occurring in a fixed interval
          when events happen at a constant average rate. It is the limit of the binomial distribution
          when N is large and p is small, with mean m = Np held constant.
          For a Poisson distribution, Mean = Variance = m.
        </p>
      </div>
    </SimulationPanel>
  );
}
