'use client';

import React, { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

function mandelbrotMask(resolution: number, maxIter: number, exponent: number): number[][] {
  const mask: number[][] = [];
  const xMin = -2.2;
  const xMax = 1.2;
  const yMin = -1.4;
  const yMax = 1.4;
  for (let i = 0; i < resolution; i++) {
    const row: number[] = [];
    const y = yMin + (i / (resolution - 1)) * (yMax - yMin);
    for (let j = 0; j < resolution; j++) {
      const x = xMin + (j / (resolution - 1)) * (xMax - xMin);
      let zr = 0;
      let zi = 0;
      let stable = 1;
      for (let n = 0; n < maxIter; n++) {
        const r = Math.sqrt(zr * zr + zi * zi);
        const theta = Math.atan2(zi, zr);
        const rPow = Math.pow(r, exponent);
        zr = rPow * Math.cos(exponent * theta) + x;
        zi = rPow * Math.sin(exponent * theta) + y;
        if (zr * zr + zi * zi > 4) {
          stable = 0;
          break;
        }
      }
      row.push(stable);
    }
    mask.push(row);
  }
  return mask;
}

function boxCountDimension(mask: number[][]): { eps: number[]; counts: number[]; slope: number } {
  const n = mask.length;
  const sizes = [2, 4, 8, 16, 32].filter(s => s <= n);
  const eps: number[] = [];
  const counts: number[] = [];
  for (const box of sizes) {
    let count = 0;
    const step = Math.floor(n / box);
    if (step < 1) continue;
    for (let i = 0; i < n; i += step) {
      for (let j = 0; j < n; j += step) {
        let has = false;
        for (let ii = i; ii < Math.min(i + step, n) && !has; ii++) {
          for (let jj = j; jj < Math.min(j + step, n); jj++) {
            if (mask[ii][jj] === 1) {
              has = true;
              break;
            }
          }
        }
        if (has) count++;
      }
    }
    eps.push(1 / box);
    counts.push(Math.max(count, 1));
  }

  const x = eps.map(v => Math.log(1 / v));
  const y = counts.map(v => Math.log(v));
  const nPts = x.length;
  let slope = 0;
  if (nPts >= 2) {
    const sx = x.reduce((s, v) => s + v, 0);
    const sy = y.reduce((s, v) => s + v, 0);
    const sxy = x.reduce((s, v, i) => s + v * y[i], 0);
    const sx2 = x.reduce((s, v) => s + v * v, 0);
    slope = (nPts * sxy - sx * sy) / (nPts * sx2 - sx * sx);
  }
  return { eps, counts, slope };
}

export function FractalDimension() {
  const [resolution, setResolution] = useState(128);
  const [maxIter, setMaxIter] = useState(35);
  const [exponent, setExponent] = useState(2.0);

  const { mask, boxes } = useMemo(() => {
    const m = mandelbrotMask(resolution, maxIter, exponent);
    return { mask: m, boxes: boxCountDimension(m) };
  }, [resolution, maxIter, exponent]);

  const darkLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(15,15,25,1)',
    font: { color: '#9ca3af' },
    margin: { t: 40, r: 20, b: 50, l: 60 },
    xaxis: { gridcolor: '#1e1e2e' },
    yaxis: { gridcolor: '#1e1e2e' },
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-6 items-center">
        <div>
          <label className="text-sm text-gray-400 block mb-1">Resolution: {resolution}</label>
          <input type="range" min={64} max={256} step={32} value={resolution} onChange={e => setResolution(Number(e.target.value))} className="w-48" />
        </div>
        <div>
          <label className="text-sm text-gray-400 block mb-1">Iterations: {maxIter}</label>
          <input type="range" min={10} max={80} step={1} value={maxIter} onChange={e => setMaxIter(Number(e.target.value))} className="w-48" />
        </div>
        <div>
          <label className="text-sm text-gray-400 block mb-1">Exponent a: {exponent.toFixed(1)}</label>
          <input type="range" min={1.5} max={4.0} step={0.1} value={exponent} onChange={e => setExponent(Number(e.target.value))} className="w-48" />
        </div>
      </div>

      <div className="text-sm text-gray-300">Estimated box-counting fractal dimension D ≈ {boxes.slope.toFixed(3)}</div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Plot
          data={[{ z: mask, type: 'heatmap', colorscale: [[0, '#0b1220'], [1, '#22d3ee']], showscale: false }]}
          layout={{
            ...darkLayout,
            title: { text: 'Mandelbrot Membership Mask', font: { size: 13, color: '#9ca3af' } },
            xaxis: { ...darkLayout.xaxis, visible: false },
            yaxis: { ...darkLayout.yaxis, visible: false },
            margin: { t: 40, r: 10, b: 10, l: 10 },
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 360 }}
        />
        <Plot
          data={[
            {
              x: boxes.eps.map(e => Math.log(1 / e)),
              y: boxes.counts.map(v => Math.log(v)),
              type: 'scatter',
              mode: 'lines+markers',
              line: { color: '#f59e0b', width: 2 },
              marker: { size: 6 },
            },
          ]}
          layout={{
            ...darkLayout,
            title: { text: 'Box Counting: log(N) vs log(1/epsilon)', font: { size: 13, color: '#9ca3af' } },
            xaxis: { ...darkLayout.xaxis, title: { text: 'log(1/epsilon)' } },
            yaxis: { ...darkLayout.yaxis, title: { text: 'log(N)' } },
            showlegend: false,
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 360 }}
        />
      </div>
    </div>
  );
}
