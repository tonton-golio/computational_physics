'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';

export default function ROCLive() {
  const [threshold, setThreshold] = useState(0.5);
  const [separation, setSeparation] = useState(1.5);

  const { rocX, rocY, auc, tpr, fpr, distX, dist0, dist1, threshVal } = useMemo(() => {
    // Two Gaussian distributions for signal and background
    const mu0 = 0;    // background
    const mu1 = separation; // signal
    const sig = 1.0;

    // Compute ROC curve analytically
    const nPoints = 200;
    const rx: number[] = [];
    const ry: number[] = [];

    const normalCDF = (x: number, mu: number, s: number) => {
      const z = (x - mu) / s;
      return 0.5 * (1 + erf(z / Math.sqrt(2)));
    };

    const erf = (x: number): number => {
      const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
      const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
      const sign = x >= 0 ? 1 : -1;
      const t = 1 / (1 + p * Math.abs(x));
      const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
      return sign * y;
    };

    for (let i = 0; i <= nPoints; i++) {
      const t = -4 + i * (separation + 8) / nPoints;
      const fp = 1 - normalCDF(t, mu0, sig); // P(score > t | background)
      const tp = 1 - normalCDF(t, mu1, sig); // P(score > t | signal)
      rx.push(fp);
      ry.push(tp);
    }

    // AUC by trapezoidal rule
    let area = 0;
    for (let i = 1; i < rx.length; i++) {
      area += 0.5 * (ry[i] + ry[i - 1]) * (rx[i - 1] - rx[i]);
    }

    // Current threshold values
    const currentFPR = 1 - normalCDF(threshold, mu0, sig);
    const currentTPR = 1 - normalCDF(threshold, mu1, sig);

    // Distribution curves for visualization
    const dx: number[] = [];
    const d0: number[] = [];
    const d1: number[] = [];
    for (let i = 0; i <= 150; i++) {
      const x = -4 + i * (separation + 8) / 150;
      dx.push(x);
      d0.push(Math.exp(-0.5 * ((x - mu0) / sig) ** 2) / (sig * Math.sqrt(2 * Math.PI)));
      d1.push(Math.exp(-0.5 * ((x - mu1) / sig) ** 2) / (sig * Math.sqrt(2 * Math.PI)));
    }

    return {
      rocX: rx, rocY: ry, auc: area,
      tpr: currentTPR, fpr: currentFPR,
      distX: dx, dist0: d0, dist1: d1, threshVal: threshold,
    };
  }, [threshold, separation]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Interactive ROC Curve</h3>
      <div className="grid grid-cols-2 gap-6 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Decision threshold: {threshold.toFixed(2)}</label>
          <Slider value={[threshold]} onValueChange={([v]) => setThreshold(v)} min={-3} max={separation + 3} step={0.05} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Class separation: {separation.toFixed(1)}</label>
          <Slider value={[separation]} onValueChange={([v]) => setSeparation(v)} min={0.2} max={4} step={0.1} />
        </div>
      </div>
      <div className="mb-3 text-sm text-[var(--text-muted)]">
        TPR: {tpr.toFixed(3)} | FPR: {fpr.toFixed(3)} | AUC: {auc.toFixed(3)}
      </div>
      <div className="grid grid-cols-2 gap-4">
        <CanvasChart
          data={[
            { x: distX, y: dist0, type: 'scatter', mode: 'lines', line: { color: '#3b82f6', width: 2 }, fill: 'tozeroy', fillcolor: 'rgba(59,130,246,0.15)', name: 'Background' },
            { x: distX, y: dist1, type: 'scatter', mode: 'lines', line: { color: '#ef4444', width: 2 }, fill: 'tozeroy', fillcolor: 'rgba(239,68,68,0.15)', name: 'Signal' },
          ]}
          layout={{
            height: 320,
            xaxis: { title: { text: 'Score' } },
            yaxis: { title: { text: 'Density' } },
            shapes: [
              { type: 'line', x0: threshVal, x1: threshVal, y0: 0, y1: 0.45, line: { color: '#10b981', width: 2, dash: 'dash' } },
            ],
          }}
          style={{ width: '100%' }}
        />
        <CanvasChart
          data={[
            { x: rocX, y: rocY, type: 'scatter', mode: 'lines', line: { color: '#3b82f6', width: 2.5 }, name: 'ROC curve' },
            { x: [fpr], y: [tpr], type: 'scatter', mode: 'markers', marker: { color: '#ef4444', size: 10 }, name: 'Current threshold' },
            { x: [0, 1], y: [0, 1], type: 'scatter', mode: 'lines', line: { color: '#94a3b8', width: 1, dash: 'dash' }, name: 'Random' },
          ]}
          layout={{
            height: 320,
            xaxis: { title: { text: 'False Positive Rate' }, range: [0, 1] },
            yaxis: { title: { text: 'True Positive Rate' }, range: [0, 1] },
          }}
          style={{ width: '100%' }}
        />
      </div>
    </div>
  );
}
