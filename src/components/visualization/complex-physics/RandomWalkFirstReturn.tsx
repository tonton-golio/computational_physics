'use client';

import React, { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

function mulberry32(seed: number) {
  let s = seed >>> 0;
  return () => {
    s += 0x6D2B79F5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function firstReturn1D(maxSteps: number, trials: number, rand: () => number): number[] {
  const out: number[] = [];
  for (let t = 0; t < trials; t++) {
    let x = 0;
    let returned = false;
    for (let step = 1; step <= maxSteps; step++) {
      x += rand() < 0.5 ? -1 : 1;
      if (x === 0) {
        out.push(step);
        returned = true;
        break;
      }
    }
    if (!returned) out.push(maxSteps + 1);
  }
  return out;
}

function firstReturn2D(maxSteps: number, trials: number, rand: () => number): number[] {
  const out: number[] = [];
  for (let t = 0; t < trials; t++) {
    let x = 0;
    let y = 0;
    let escaped = false;
    let returned = false;
    for (let step = 1; step <= maxSteps; step++) {
      const theta = rand() * 2 * Math.PI;
      x += Math.cos(theta);
      y += Math.sin(theta);
      const r = Math.sqrt(x * x + y * y);
      if (r > 1.5) escaped = true;
      if (escaped && r <= 1.0) {
        out.push(step);
        returned = true;
        break;
      }
    }
    if (!returned) out.push(maxSteps + 1);
  }
  return out;
}

function randomWalkPath2D(steps: number, rand: () => number): { x: number[]; y: number[] } {
  const x = [0];
  const y = [0];
  for (let i = 0; i < steps; i++) {
    const theta = rand() * 2 * Math.PI;
    x.push(x[x.length - 1] + Math.cos(theta));
    y.push(y[y.length - 1] + Math.sin(theta));
  }
  return { x, y };
}

export function RandomWalkFirstReturn() {
  const [maxSteps, setMaxSteps] = useState(1200);
  const [trials, setTrials] = useState(700);
  const [pathSteps, setPathSteps] = useState(240);
  const [rerun, setRerun] = useState(0);

  const { ret1d, ret2d, path } = useMemo(() => {
    const rand = mulberry32((maxSteps * 17 + trials * 23 + pathSteps * 31 + rerun * 101) >>> 0);
    const a = firstReturn1D(maxSteps, trials, rand);
    const b = firstReturn2D(maxSteps, trials, rand);
    const c = randomWalkPath2D(pathSteps, rand);
    return { ret1d: a, ret2d: b, path: c };
  }, [maxSteps, trials, pathSteps, rerun]);

  const cutoff = maxSteps + 1;
  const finite1d = ret1d.filter(v => v < cutoff);
  const finite2d = ret2d.filter(v => v < cutoff);

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
          <label className="text-sm text-gray-400 block mb-1">Max Steps: {maxSteps}</label>
          <input type="range" min={200} max={3000} step={100} value={maxSteps} onChange={e => setMaxSteps(Number(e.target.value))} className="w-48" />
        </div>
        <div>
          <label className="text-sm text-gray-400 block mb-1">Trials: {trials}</label>
          <input type="range" min={100} max={3000} step={100} value={trials} onChange={e => setTrials(Number(e.target.value))} className="w-48" />
        </div>
        <div>
          <label className="text-sm text-gray-400 block mb-1">Path Steps: {pathSteps}</label>
          <input type="range" min={50} max={1200} step={50} value={pathSteps} onChange={e => setPathSteps(Number(e.target.value))} className="w-48" />
        </div>
        <button onClick={() => setRerun(v => v + 1)} className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-sm mt-4">
          Re-run
        </button>
      </div>

      <div className="text-sm text-gray-300">
        Returned before cutoff: 1D {finite1d.length}/{trials} | 2D {finite2d.length}/{trials}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Plot
          data={[{ x: finite1d, type: 'histogram', marker: { color: '#60a5fa' }, nbinsx: 40 }]}
          layout={{
            ...darkLayout,
            title: { text: 'First Return Distribution (1D)', font: { size: 13, color: '#9ca3af' } },
            xaxis: { ...darkLayout.xaxis, title: { text: 'Return Step' } },
            yaxis: { ...darkLayout.yaxis, title: { text: 'Count' } },
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 320 }}
        />
        <Plot
          data={[{ x: finite2d, type: 'histogram', marker: { color: '#34d399' }, nbinsx: 40 }]}
          layout={{
            ...darkLayout,
            title: { text: 'First Return Distribution (2D)', font: { size: 13, color: '#9ca3af' } },
            xaxis: { ...darkLayout.xaxis, title: { text: 'Return Step' } },
            yaxis: { ...darkLayout.yaxis, title: { text: 'Count' } },
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 320 }}
        />
        <Plot
          data={[
            { x: path.x, y: path.y, type: 'scatter', mode: 'lines', line: { color: '#f59e0b', width: 1.5 } },
            { x: [0], y: [0], type: 'scatter', mode: 'markers', marker: { color: '#ef4444', size: 8 } },
          ]}
          layout={{
            ...darkLayout,
            title: { text: 'Sample 2D Random Walk Path', font: { size: 13, color: '#9ca3af' } },
            xaxis: { ...darkLayout.xaxis, title: { text: 'x' } },
            yaxis: { ...darkLayout.yaxis, title: { text: 'y' }, scaleanchor: 'x', scaleratio: 1 },
            showlegend: false,
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 320 }}
        />
      </div>
    </div>
  );
}
