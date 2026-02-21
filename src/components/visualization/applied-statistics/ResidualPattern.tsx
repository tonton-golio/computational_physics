'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';

export default function ResidualPattern() {
  const [curvature, setCurvature] = useState(0.0);
  const [noise, setNoise] = useState(0.5);

  const { xVals, yVals, fitY, residuals } = useMemo(() => {
    const N = 60;
    const xs: number[] = [];
    const ys: number[] = [];
    // True model: y = 2 + 1.5x + curvature * x^2
    for (let i = 0; i < N; i++) {
      const x = i / (N - 1) * 10;
      xs.push(x);
      const y = 2 + 1.5 * x + curvature * x * x
        + (Math.sin(i * 1.37) * 2 - 1) * noise;
      ys.push(y);
    }
    // Fit a straight line (always)
    const n = xs.length;
    const sx = xs.reduce((a, b) => a + b, 0);
    const sy = ys.reduce((a, b) => a + b, 0);
    const sxx = xs.reduce((a, x) => a + x * x, 0);
    const sxy = xs.reduce((a, x, i) => a + x * ys[i], 0);
    const b1 = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    const b0 = (sy - b1 * sx) / n;
    const fy = xs.map((x) => b0 + b1 * x);
    const res = ys.map((y, i) => y - fy[i]);
    return { xVals: xs, yVals: ys, fitY: fy, residuals: res };
  }, [curvature, noise]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Residual Patterns: Good Fit vs Bad Fit</h3>
      <div className="grid grid-cols-2 gap-6 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Curvature (quadratic term): {curvature.toFixed(2)}</label>
          <Slider value={[curvature]} onValueChange={([v]) => setCurvature(v)} min={0} max={0.5} step={0.01} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Noise: {noise.toFixed(2)}</label>
          <Slider value={[noise]} onValueChange={([v]) => setNoise(v)} min={0.1} max={3} step={0.1} />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <CanvasChart
          data={[
            { x: xVals, y: yVals, type: 'scatter', mode: 'markers', marker: { color: '#3b82f6', size: 4 }, name: 'Data' },
            { x: xVals, y: fitY, type: 'scatter', mode: 'lines', line: { color: '#ef4444', width: 2 }, name: 'Linear fit' },
          ]}
          layout={{
            height: 320,
            xaxis: { title: { text: 'x' } },
            yaxis: { title: { text: 'y' } },
          }}
          style={{ width: '100%' }}
        />
        <CanvasChart
          data={[
            { x: xVals, y: residuals, type: 'scatter', mode: 'markers', marker: { color: '#8b5cf6', size: 4 }, name: 'Residuals' },
          ]}
          layout={{
            height: 320,
            xaxis: { title: { text: 'x' } },
            yaxis: { title: { text: 'Residual (y - fit)' } },
            shapes: [
              { type: 'line', x0: 0, x1: 10, y0: 0, y1: 0, line: { color: '#94a3b8', width: 1, dash: 'dash' } },
            ],
          }}
          style={{ width: '100%' }}
        />
      </div>
    </div>
  );
}
