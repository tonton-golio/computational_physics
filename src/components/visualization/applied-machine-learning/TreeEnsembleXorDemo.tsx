"use client";

import React, { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';
import { pseudoRandom, gaussianNoise } from './ml-utils';

function xorData(n: number, noise: number): { x1: number[]; x2: number[]; label: number[] } {
  const x1: number[] = [];
  const x2: number[] = [];
  const label: number[] = [];
  for (let i = 0; i < n; i++) {
    const a = pseudoRandom(i * 1.7 + 5) > 0.5 ? 1 : 0;
    const b = pseudoRandom(i * 2.1 + 11) > 0.5 ? 1 : 0;
    const y = a === b ? 0 : 1;
    x1.push(a + gaussianNoise(i * 3.1 + 7, noise));
    x2.push(b + gaussianNoise(i * 4.3 + 13, noise));
    label.push(y);
  }
  return { x1, x2, label };
}

export default function TreeEnsembleXorDemo({}: SimulationComponentProps): React.ReactElement {
  const [n, setN] = useState(400);
  const [noise, setNoise] = useState(0.2);
  const [threshold, setThreshold] = useState(0.6);
  const data = useMemo(() => xorData(n, noise), [n, noise]);
  const probs = useMemo(() => data.x1.map((a, i) => 0.5 + 0.45 * Math.sin(3 * (a - 0.5) * (data.x2[i] - 0.5))), [data]);
  const pred = probs.map((p) => (p > threshold ? 1 : 0));

  // Map numeric label arrays to color strings using Portland colorscale approximation
  const groundTruthColors = data.label.map((l) => (l === 0 ? '#3b82f6' : '#ef4444'));
  const predColors = pred.map((l) => (l === 0 ? '#3b82f6' : '#ef4444'));

  return (
    <SimulationPanel title="XOR Classification Intuition">
      <SimulationConfig>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <div>
            <SimulationLabel>Samples: {n}</SimulationLabel>
            <Slider min={100} max={1000} step={50} value={[n]} onValueChange={([v]) => setN(v)} />
          </div>
          <div>
            <SimulationLabel>Noise: {noise.toFixed(2)}</SimulationLabel>
            <Slider min={0.05} max={0.45} step={0.01} value={[noise]} onValueChange={([v]) => setNoise(v)} />
          </div>
          <div>
            <SimulationLabel>Decision threshold: {threshold.toFixed(2)}</SimulationLabel>
            <Slider min={0.3} max={0.8} step={0.01} value={[threshold]} onValueChange={([v]) => setThreshold(v)} />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div>
            <p className="mb-1 text-center text-sm text-[var(--text-muted)]">Ground truth</p>
            <CanvasChart
              data={[
                {
                  x: data.x1,
                  y: data.x2,
                  mode: 'markers',
                  type: 'scatter',
                  name: 'Ground truth',
                  marker: { color: groundTruthColors, size: 6, opacity: 0.55 },
                },
              ]}
              layout={{
                margin: { t: 20, r: 20, b: 40, l: 45 },
                showlegend: false,
              }}
              style={{ width: '100%', height: 360 }}
            />
          </div>
          <div>
            <p className="mb-1 text-center text-sm text-[var(--text-muted)]">Predicted</p>
            <CanvasChart
              data={[
                {
                  x: data.x1,
                  y: data.x2,
                  mode: 'markers',
                  type: 'scatter',
                  name: 'Predicted',
                  marker: { color: predColors, size: 6, opacity: 0.55 },
                },
              ]}
              layout={{
                margin: { t: 20, r: 20, b: 40, l: 45 },
                showlegend: false,
              }}
              style={{ width: '100%', height: 360 }}
            />
          </div>
        </div>
      </SimulationMain>
    </SimulationPanel>
  );
}
