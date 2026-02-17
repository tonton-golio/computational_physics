'use client';

import React, { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id: string;
}

// Box-Muller transform for generating normal random variables
function boxMuller(): number {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Chi-squared distribution demo.
 * Generates noisy linear data y = a*x + b, computes chi2 over a grid of (a,b),
 * then shows the data + fit alongside a chi2 surface heatmap.
 */
export default function AppliedStatsSim5({ }: SimulationProps) {
  const [resolution, setResolution] = useState(40);
  const [nSamples, setNSamples] = useState(15);
  const [seed, setSeed] = useState(42);

  const result = useMemo(() => {
    // Seeded pseudo-random using a simple LCG for reproducibility
    let s = seed;
    const seededRandom = () => {
      s = (s * 1664525 + 1013904223) % 4294967296;
      return s / 4294967296;
    };
    // Box-Muller with seeded random
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
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">Chi-Squared Grid Search Demo</h3>
      <p className="text-sm text-gray-300 mb-4">
        Generate noisy data from y = 2x + 4, then search over a grid of (a, b) values to find the parameters
        that minimize the chi-squared statistic. The left plot shows data and the best fit; the right plot
        shows the chi-squared surface as a heatmap.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div>
          <label className="text-sm text-gray-400">Resolution: {resolution}</label>
          <input type="range" min={10} max={80} step={5} value={resolution}
            onChange={e => setResolution(+e.target.value)} className="w-full" />
        </div>
        <div>
          <label className="text-sm text-gray-400">Data Points: {nSamples}</label>
          <input type="range" min={5} max={40} step={1} value={nSamples}
            onChange={e => setNSamples(+e.target.value)} className="w-full" />
        </div>
        <div>
          <label className="text-sm text-gray-400">Seed: {seed}</label>
          <input type="range" min={1} max={200} step={1} value={seed}
            onChange={e => setSeed(+e.target.value)} className="w-full" />
        </div>
      </div>

      <div className="text-sm text-gray-300 mb-2">
        Best fit: a = {result.bestA.toFixed(3)}, b = {result.bestB.toFixed(3)} | RMSE = {result.rmse.toFixed(4)}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Plot
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
            title: { text: 'Data and Best Fit', font: { color: '#fff' } },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(15,15,25,1)',
            font: { color: '#9ca3af' },
            margin: { t: 40, r: 20, b: 50, l: 50 },
            xaxis: { title: { text: 'x' }, gridcolor: '#1e1e2e' },
            yaxis: { title: { text: 'y' }, gridcolor: '#1e1e2e' },
            legend: { font: { color: '#9ca3af' } },
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 400 }}
        />
        <Plot
          data={[
            {
              z: result.Z,
              x: result.bList,
              y: result.aList,
              type: 'heatmap',
              colorscale: 'Hot',
              reversescale: true,
              colorbar: { title: { text: '\u03C7\u00B2', side: 'right' }, tickfont: { color: '#9ca3af' } },
            },
          ]}
          layout={{
            title: { text: '\u03C7\u00B2 Surface', font: { color: '#fff' } },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(15,15,25,1)',
            font: { color: '#9ca3af' },
            margin: { t: 40, r: 80, b: 50, l: 50 },
            xaxis: { title: { text: 'b' }, gridcolor: '#1e1e2e' },
            yaxis: { title: { text: 'a' }, gridcolor: '#1e1e2e' },
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 400 }}
        />
      </div>
    </div>
  );
}
