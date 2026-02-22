"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

const MEASUREMENT_VALUES = [10.2, 10.5, 10.1];

export default function WeightedMean({}: SimulationComponentProps) {
  const [sigma1, setSigma1] = useState(0.1);
  const [sigma2, setSigma2] = useState(0.5);
  const [sigma3, setSigma3] = useState(0.2);

  const values = MEASUREMENT_VALUES;

  const { wMean, wSigma, weights, arithMean } = useMemo(() => {
    const sigmas = [sigma1, sigma2, sigma3];
    const w = sigmas.map((s) => 1 / (s * s));
    const sumW = w.reduce((a, b) => a + b, 0);
    const wm = MEASUREMENT_VALUES.reduce((acc, v, i) => acc + v * w[i], 0) / sumW;
    const ws = Math.sqrt(1 / sumW);
    const am = MEASUREMENT_VALUES.reduce((a, b) => a + b, 0) / MEASUREMENT_VALUES.length;
    return { wMean: wm, wSigma: ws, weights: w.map((wi) => wi / sumW), arithMean: am };
  }, [sigma1, sigma2, sigma3]);

  const sigmas = [sigma1, sigma2, sigma3];
  const colors = ['#3b82f6', '#f59e0b', '#10b981'];
  const markers: Array<{ x: number[]; y: number[]; type: 'scatter'; mode: string; marker: { color: string; size: number }; name: string }> = values.map((v, i) => ({
    x: [v],
    y: [i],
    type: 'scatter' as const,
    mode: 'markers',
    marker: { color: colors[i], size: 10 },
    name: `x${i + 1} = ${v} (w=${weights[i].toFixed(2)})`,
  }));

  const errBars = values.map((v, i) => ({
    x: [v - sigmas[i], v + sigmas[i]],
    y: [i, i],
    type: 'scatter' as const,
    mode: 'lines' as const,
    line: { color: colors[i], width: 3 },
    showlegend: false,
  }));

  return (
    <SimulationPanel title="Weighted Mean: Precision Matters">
      <SimulationConfig>
        <div className="grid grid-cols-3 gap-6">
          <div>
            <SimulationLabel>sigma1: {sigma1.toFixed(2)}</SimulationLabel>
            <Slider value={[sigma1]} onValueChange={([v]) => setSigma1(v)} min={0.05} max={1.0} step={0.01} />
          </div>
          <div>
            <SimulationLabel>sigma2: {sigma2.toFixed(2)}</SimulationLabel>
            <Slider value={[sigma2]} onValueChange={([v]) => setSigma2(v)} min={0.05} max={1.0} step={0.01} />
          </div>
          <div>
            <SimulationLabel>sigma3: {sigma3.toFixed(2)}</SimulationLabel>
            <Slider value={[sigma3]} onValueChange={([v]) => setSigma3(v)} min={0.05} max={1.0} step={0.01} />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
      <CanvasChart
        data={[
          ...errBars as any,
          ...markers as any,
        ]}
        layout={{
          height: 280,
          xaxis: { title: { text: 'Measurement value' }, range: [9.0, 11.5] },
          yaxis: { title: { text: '' }, range: [-0.8, 2.8], showgrid: false },
          shapes: [
            { type: 'line', x0: wMean, x1: wMean, y0: -0.6, y1: 2.6, line: { color: '#ef4444', width: 2.5, dash: 'solid' } },
            { type: 'line', x0: arithMean, x1: arithMean, y0: -0.6, y1: 2.6, line: { color: '#94a3b8', width: 2, dash: 'dash' } },
          ],
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>
      <SimulationResults>
        <div className="text-sm text-[var(--text-muted)] flex flex-wrap gap-4">
          <span>Arithmetic mean: {arithMean.toFixed(3)}</span>
          <span>Weighted mean: {wMean.toFixed(3)} +/- {wSigma.toFixed(3)}</span>
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
