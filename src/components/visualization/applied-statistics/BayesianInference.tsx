"use client";

import { useState, useMemo } from 'react';
import { mulberry32 } from '@/lib/math';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';


/**
 * Chi-squared distribution demo.
 * Generates noisy linear data y = a*x + b, computes chi2 over a grid of (a,b),
 * then shows the data + fit alongside a chi2 surface heatmap.
 */
export default function AppliedStatsSim5({}: SimulationComponentProps) {
  const [resolution, setResolution] = useState(40);
  const [nSamples, setNSamples] = useState(15);
  const [seed, setSeed] = useState(42);

  const result = useMemo(() => {
    const seededRandom = mulberry32(seed);
    const seededNormal = () => {
      const u1 = seededRandom();
      const u2 = seededRandom();
      return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    };

    const aTrue = 2;
    const bTrue = 4;

    // Generate data: y = a*x + b + noise
    const x: number[] = [];
    const yGt: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      const xi = -1 + (2 * i) / (nSamples - 1);
      x.push(xi);
      yGt.push(aTrue * xi + bTrue + seededNormal() * 0.3);
    }

    // Chi2 grid search
    const aList: number[] = [];
    const bList: number[] = [];
    for (let i = 0; i < resolution; i++) {
      aList.push(-2 + (7 * i) / (resolution - 1));
      bList.push(2 + (4 * i) / (resolution - 1));
    }

    const Z: number[][] = [];
    for (let i = 0; i < resolution; i++) {
      const row: number[] = [];
      for (let j = 0; j < resolution; j++) {
        const a = aList[i];
        const b = bList[j];
        let chi2 = 0;
        for (let k = 0; k < nSamples; k++) {
          const predicted = a * x[k] + b;
          const diff = predicted - yGt[k];
          chi2 += (diff * diff) / Math.max(Math.abs(yGt[k]), 0.01);
        }
        row.push(chi2);
      }
      Z.push(row);
    }

    // Find optimal (a, b) from grid
    let minChi2 = Infinity;
    let bestA = aList[0];
    let bestB = bList[0];
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        if (Z[i][j] < minChi2) {
          minChi2 = Z[i][j];
          bestA = aList[i];
          bestB = bList[j];
        }
      }
    }

    // Fitted line
    const xFit: number[] = [];
    const yFit: number[] = [];
    for (let i = 0; i <= 50; i++) {
      const xi = -1 + (2 * i) / 50;
      xFit.push(xi);
      yFit.push(bestA * xi + bestB);
    }

    // RMSE
    let sse = 0;
    for (let k = 0; k < nSamples; k++) {
      const diff = yGt[k] - (bestA * x[k] + bestB);
      sse += diff * diff;
    }
    const rmse = Math.sqrt(sse);

    return { x, yGt, xFit, yFit, aList, bList, Z, bestA, bestB, rmse };
  }, [resolution, nSamples, seed]);

  return (
    <SimulationPanel title="Chi-Squared Grid Search Demo" caption="Generate noisy data from y = 2x + 4, then search over a grid of (a, b) values to find the parameters that minimize the chi-squared statistic. The left plot shows data and the best fit; the right plot shows the chi-squared surface as a heatmap.">
      <SimulationConfig>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <SimulationLabel>Resolution: {resolution}</SimulationLabel>
            <Slider min={10} max={80} step={5} value={[resolution]}
              onValueChange={([v]) => setResolution(v)} />
          </div>
          <div>
            <SimulationLabel>Data Points: {nSamples}</SimulationLabel>
            <Slider min={5} max={40} step={1} value={[nSamples]}
              onValueChange={([v]) => setNSamples(v)} />
          </div>
          <div>
            <SimulationLabel>Seed: {seed}</SimulationLabel>
            <Slider min={1} max={200} step={1} value={[seed]}
              onValueChange={([v]) => setSeed(v)} />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <CanvasChart
            data={[
              {
                x: result.x,
                y: result.yGt,
                type: 'scatter',
                mode: 'markers',
                marker: { color: 'cyan', size: 8, symbol: 'x' },
                name: 'Data',
              },
              {
                x: result.xFit,
                y: result.yFit,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#ef4444', width: 2 },
                name: 'Best fit',
              },
            ]}
            layout={{
              title: { text: 'Data and Best Fit' },
              margin: { t: 40, r: 20, b: 50, l: 50 },
              xaxis: { title: { text: 'x' } },
              yaxis: { title: { text: 'y' } },
              legend: {},
            }}
            style={{ width: '100%', height: 400 }}
          />
          <CanvasHeatmap
            data={[
              {
                z: result.Z,
                x: result.bList,
                y: result.aList,
                type: 'heatmap',
                colorscale: 'Hot',
                reversescale: true,
                colorbar: { title: { text: '\u03C7\u00B2', side: 'right' } },
              },
            ]}
            layout={{
              title: { text: '\u03C7\u00B2 Surface' },
              margin: { t: 40, r: 80, b: 50, l: 50 },
              xaxis: { title: { text: 'b' } },
              yaxis: { title: { text: 'a' } },
            }}
            style={{ width: '100%', height: 400 }}
          />
        </div>
      </SimulationMain>
      <SimulationResults>
        <div className="text-sm text-[var(--text-muted)]">
          Best fit: a = {result.bestA.toFixed(3)}, b = {result.bestB.toFixed(3)} | RMSE = {result.rmse.toFixed(4)}
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
