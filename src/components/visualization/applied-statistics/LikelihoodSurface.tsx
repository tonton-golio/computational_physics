"use client";

import { useState, useMemo, useCallback } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationSettings, SimulationResults, SimulationButton } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

export default function LikelihoodSurface({}: SimulationComponentProps) {
  const [dataPoints, setDataPoints] = useState<number[]>([2.1, 3.5, 2.8, 4.0, 3.2]);

  const { contourTraces, peakMu, peakSig } = useMemo(() => {
    const nMu = 60;
    const nSig = 60;
    const muMin = 0;
    const muMax = 6;
    const sigMin = 0.3;
    const sigMax = 4;
    const muStep = (muMax - muMin) / nMu;
    const sigStep = (sigMax - sigMin) / nSig;

    const mus: number[] = [];
    const sigs: number[] = [];
    for (let i = 0; i <= nMu; i++) mus.push(muMin + i * muStep);
    for (let j = 0; j <= nSig; j++) sigs.push(sigMin + j * sigStep);

    // Compute log-likelihood on grid
    const grid: number[][] = [];
    let maxLL = -Infinity;
    let bestMu = 3;
    let bestSig = 1;
    for (let j = 0; j <= nSig; j++) {
      const row: number[] = [];
      for (let i = 0; i <= nMu; i++) {
        let ll = 0;
        for (const x of dataPoints) {
          const z = (x - mus[i]) / sigs[j];
          ll += -0.5 * z * z - Math.log(sigs[j]) - 0.5 * Math.log(2 * Math.PI);
        }
        row.push(ll);
        if (ll > maxLL) { maxLL = ll; bestMu = mus[i]; bestSig = sigs[j]; }
      }
      grid.push(row);
    }

    // Extract contour levels as scatter traces
    const levels = [0.5, 2, 4.5, 8];
    const traces: any[] = [];
    const contourColors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'];
    for (let li = 0; li < levels.length; li++) {
      const threshold = maxLL - levels[li];
      const xs: number[] = [];
      const ys: number[] = [];
      for (let j = 0; j <= nSig; j++) {
        for (let i = 0; i <= nMu; i++) {
          if (grid[j][i] >= threshold) {
            xs.push(mus[i]);
            ys.push(sigs[j]);
          }
        }
      }
      traces.push({
        x: xs, y: ys, type: 'scatter', mode: 'markers',
        marker: { color: contourColors[li], size: 3, opacity: 0.4 },
        name: `-2DeltaLL < ${levels[li] * 2}`,
      });
    }

    // MLE marker
    traces.push({
      x: [bestMu], y: [bestSig], type: 'scatter', mode: 'markers',
      marker: { color: '#ffffff', size: 8, line: { width: 2, color: '#ef4444' } },
      name: `MLE (mu=${bestMu.toFixed(2)}, sig=${bestSig.toFixed(2)})`,
    });

    return { contourTraces: traces, peakMu: bestMu, peakSig: bestSig };
  }, [dataPoints]);

  const addPoint = useCallback(() => {
    const newVal = 1 + Math.random() * 4;
    setDataPoints((prev) => [...prev, parseFloat(newVal.toFixed(2))]);
  }, []);

  const removePoint = useCallback(() => {
    setDataPoints((prev) => (prev.length > 1 ? prev.slice(0, -1) : prev));
  }, []);

  return (
    <SimulationPanel title="Likelihood Surface (Gaussian mu, sigma)">
      <SimulationSettings>
        <SimulationButton variant="primary" onClick={addPoint}>+ Add data point</SimulationButton>
        <SimulationButton onClick={removePoint}>- Remove</SimulationButton>
        <span className="text-sm text-[var(--text-muted)] self-center">Data: [{dataPoints.map((d) => d.toFixed(1)).join(', ')}]</span>
      </SimulationSettings>
      <SimulationMain>
      <CanvasChart
        data={contourTraces}
        layout={{
          height: 420,
          xaxis: { title: { text: 'mu (mean)' }, range: [0, 6] },
          yaxis: { title: { text: 'sigma (std dev)' }, range: [0.3, 4] },
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>
      <SimulationResults>
        <div className="text-sm text-[var(--text-muted)]">
          MLE: mu = {peakMu.toFixed(2)}, sigma = {peakSig.toFixed(2)} | n = {dataPoints.length}
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
