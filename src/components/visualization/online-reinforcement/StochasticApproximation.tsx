"use client";

import { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationLabel, SimulationButton, SimulationSettings, SimulationConfig } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';


function mulberry32(a: number) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export default function StochasticApproximation({}: SimulationComponentProps) {
  const [steps, setSteps] = useState(300);
  const [alpha, setAlpha] = useState(0.03);
  const [noiseScale, setNoiseScale] = useState(0.1);
  const [seed, setSeed] = useState(7);

  const { xs, ys } = useMemo(() => {
    const rng = mulberry32(seed);
    const xVals: number[] = [0.6];
    const yVals: number[] = [Math.abs(xVals[0] * xVals[0] - 1)];
    for (let k = 1; k <= steps; k++) {
      const prev = xVals[k - 1];
      const noise = (rng() - 0.5) * 2 * noiseScale;
      const gradientProxy = (prev * prev - 1) + noise;
      const next = prev - alpha * gradientProxy;
      xVals.push(next);
      yVals.push(Math.abs(next * next - 1));
    }
    return { xs: xVals, ys: yVals };
  }, [steps, alpha, noiseScale, seed]);

  return (
    <SimulationPanel title="Robbins-Monro Stochastic Approximation">
      <SimulationSettings>
        <SimulationButton variant="primary" onClick={() => setSeed((s) => s + 1)}>Re-sample</SimulationButton>
      </SimulationSettings>
      <SimulationConfig>
        <div>
          <SimulationLabel>Steps: {steps}</SimulationLabel>
          <Slider value={[steps]} onValueChange={([v]) => setSteps(v)} min={50} max={1000} step={10} />
        </div>
        <div>
          <SimulationLabel>Step size: {alpha.toFixed(3)}</SimulationLabel>
          <Slider value={[alpha]} onValueChange={([v]) => setAlpha(v)} min={0.005} max={0.1} step={0.005} />
        </div>
        <div>
          <SimulationLabel>Noise: {noiseScale.toFixed(2)}</SimulationLabel>
          <Slider value={[noiseScale]} onValueChange={([v]) => setNoiseScale(v)} min={0} max={0.5} step={0.01} />
        </div>
      </SimulationConfig>
      <SimulationMain>
      <CanvasChart
        data={[
          { x: Array.from({ length: xs.length }, (_, i) => i), y: xs, type: 'scatter', mode: 'lines', name: 'x_k', line: { color: '#60a5fa', width: 2 } },
          { x: Array.from({ length: ys.length }, (_, i) => i), y: ys.map((v) => Math.log10(v + 1e-6)), type: 'scatter', mode: 'lines', name: 'log10 residual', line: { color: '#facc15', width: 2 } },
        ]}
        layout={{
          title: { text: 'Convergence behavior under noisy updates' },
          xaxis: { title: { text: 'iteration k' } },
          yaxis: { title: { text: 'state / log residual' } },
          height: 420,
          margin: { t: 40, b: 60, l: 60, r: 60 },
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>
    </SimulationPanel>
  );
}
