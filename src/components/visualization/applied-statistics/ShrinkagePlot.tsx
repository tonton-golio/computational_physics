'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';

export default function ShrinkagePlot() {
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
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Bayesian Shrinkage</h3>
      <div className="grid grid-cols-1 gap-6 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Prior strength (tau): {priorStrength.toFixed(1)}</label>
          <Slider value={[priorStrength]} onValueChange={([v]) => setPriorStrength(v)} min={0.1} max={10} step={0.1} />
        </div>
      </div>
      <div className="mb-3 text-sm text-[var(--text-muted)]">
        Grand mean: {grandMean.toFixed(1)} | Small tau = strong shrinkage toward grand mean | Large tau = minimal shrinkage
      </div>
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
    </div>
  );
}
