"use client";

import { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

function factorial(n: number): number {
  if (n <= 1) return 1;
  let out = 1;
  for (let i = 2; i <= n; i++) out *= i;
  return out;
}

function choose(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  return factorial(n) / (factorial(k) * factorial(n - k));
}

export default function BinomialPoissonComparison({}: SimulationComponentProps) {
  const [n, setN] = useState(40);
  const [mean, setMean] = useState(6);

  const { ks, binom, poisson } = useMemo(() => {
    const p = mean / n;
    const maxK = Math.max(20, Math.ceil(mean * 2.8));
    const ks = Array.from({ length: maxK + 1 }, (_, i) => i);
    const binom = ks.map((k) => choose(n, k) * p ** k * (1 - p) ** (n - k));
    const poisson = ks.map((k) => Math.exp(-mean) * mean ** k / factorial(k));
    return { ks, binom, poisson };
  }, [n, mean]);

  return (
    <SimulationPanel title="Binomial vs Poisson Limit" caption="Compare exact Binomial(n, p) with Poisson(lambda) where lambda = n p.">
      <SimulationConfig>
        <div>
          <SimulationLabel>n (trials): {n}</SimulationLabel>
          <Slider value={[n]} onValueChange={([v]) => setN(v)} min={10} max={200} step={5} />
        </div>
        <div>
          <SimulationLabel>lambda = n p: {mean.toFixed(1)}</SimulationLabel>
          <Slider value={[mean]} onValueChange={([v]) => setMean(v)} min={1} max={20} step={0.5} />
        </div>
      </SimulationConfig>
      <SimulationMain>
      <CanvasChart
        data={[
          {
            x: ks,
            y: binom,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#3b82f6', width: 2 },
            marker: { size: 5 },
            name: 'Binomial',
          },
          {
            x: ks,
            y: poisson,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#ec4899', width: 2, dash: 'dot' },
            marker: { size: 5 },
            name: 'Poisson approximation',
          },
        ]}
        layout={{
          title: { text: 'Distribution comparison' },
          margin: { t: 40, r: 20, b: 40, l: 50 },
          xaxis: { title: { text: 'k events' } },
          yaxis: { title: { text: 'Probability' } },
          height: 430,
        }}
        style={{ width: '100%', height: 430 }}
      />
      </SimulationMain>
    </SimulationPanel>
  );
}
