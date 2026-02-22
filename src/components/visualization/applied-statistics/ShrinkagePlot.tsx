"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

export default function ShrinkagePlot({}: SimulationComponentProps) {
  const [priorStrength, setPriorStrength] = useState(2.0);

  const { rawMeans, shrunkMeans, groupLabels, grandMean } = useMemo(() => {
    // 8 groups with varying sample sizes and raw means
    const raw = [12.5, 8.3, 15.1, 6.7, 11.2, 9.8, 14.0, 7.5];
    const ns = [5, 20, 3, 15, 8, 25, 4, 10];
    const withinVar = 10; // sigma^2_within
    const gm = raw.reduce((a, b) => a + b, 0) / raw.length;
    const tauSq = priorStrength * priorStrength;

    const shrunk = raw.map((m, i) => {
      const B = withinVar / ns[i] / (withinVar / ns[i] + tauSq);
      return m * (1 - B) + gm * B;
    });

    const labels = raw.map((_, i) => `G${i + 1} (n=${ns[i]})`);
    return { rawMeans: raw, shrunkMeans: shrunk, groupLabels: labels, grandMean: gm };
  }, [priorStrength]);

  // Build arrow-like traces: from raw to shrunk
  const arrowTraces: any[] = rawMeans.map((r, i) => ({
    x: [r, shrunkMeans[i]],
    y: [i, i],
    type: 'scatter',
    mode: 'lines',
    line: { color: '#94a3b8', width: 1.5 },
    showlegend: false,
  }));

  return (
    <SimulationPanel title="Bayesian Shrinkage">
      <SimulationConfig>
        <div>
          <SimulationLabel>Prior strength (tau): {priorStrength.toFixed(1)}</SimulationLabel>
          <Slider value={[priorStrength]} onValueChange={([v]) => setPriorStrength(v)} min={0.1} max={10} step={0.1} />
        </div>
      </SimulationConfig>
      <SimulationMain>
      <CanvasChart
        data={[
          ...arrowTraces,
          {
            x: rawMeans, y: rawMeans.map((_, i) => i),
            type: 'scatter', mode: 'markers',
            marker: { color: '#3b82f6', size: 8 }, name: 'Raw means',
          },
          {
            x: shrunkMeans, y: shrunkMeans.map((_, i) => i),
            type: 'scatter', mode: 'markers',
            marker: { color: '#ef4444', size: 8 }, name: 'Shrunk means',
          },
        ]}
        layout={{
          height: 400,
          xaxis: { title: { text: 'Group mean estimate' } },
          yaxis: {
            title: { text: '' },
            tickvals: rawMeans.map((_, i) => i),
            ticktext: groupLabels,
          },
          shapes: [
            {
              type: 'line', x0: grandMean, x1: grandMean, y0: -0.5, y1: rawMeans.length - 0.5,
              line: { color: '#10b981', width: 2, dash: 'dash' },
            },
          ],
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>
      <SimulationResults>
        <div className="text-sm text-[var(--text-muted)]">
          Grand mean: {grandMean.toFixed(1)} | Small tau = strong shrinkage toward grand mean | Large tau = minimal shrinkage
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
