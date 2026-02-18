'use client';

import React, { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';
import { Slider } from '@/components/ui/slider';
import { usePlotlyTheme } from '@/lib/plotly-theme';

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
  const { mergeLayout } = usePlotlyTheme();

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

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-6 items-center">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Max Steps: {maxSteps}</label>
          <Slider value={[maxSteps]} onValueChange={([v]) => setMaxSteps(v)} min={200} max={3000} step={100} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Trials: {trials}</label>
          <Slider value={[trials]} onValueChange={([v]) => setTrials(v)} min={100} max={3000} step={100} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Path Steps: {pathSteps}</label>
          <Slider value={[pathSteps]} onValueChange={([v]) => setPathSteps(v)} min={50} max={1200} step={50} />
        </div>
        <button onClick={() => setRerun(v => v + 1)} className="px-4 py-2 bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white rounded text-sm mt-4">
          Re-run
        </button>
      </div>

      <div className="text-sm text-[var(--text-muted)]">
        Returned before cutoff: 1D {finite1d.length}/{trials} | 2D {finite2d.length}/{trials}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Plot
          data={[{ x: finite1d, type: 'histogram', marker: { color: '#60a5fa' }, nbinsx: 40 }]}
          layout={mergeLayout({
            title: { text: 'First Return Distribution (1D)', font: { size: 13 } },
            xaxis: { title: { text: 'Return Step' } },
            yaxis: { title: { text: 'Count' } },
            margin: { t: 40, r: 20, b: 50, l: 60 },
          })}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 320 }}
        />
        <Plot
          data={[{ x: finite2d, type: 'histogram', marker: { color: '#34d399' }, nbinsx: 40 }]}
          layout={mergeLayout({
            title: { text: 'First Return Distribution (2D)', font: { size: 13 } },
            xaxis: { title: { text: 'Return Step' } },
            yaxis: { title: { text: 'Count' } },
            margin: { t: 40, r: 20, b: 50, l: 60 },
          })}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 320 }}
        />
        <Plot
          data={[
            { x: path.x, y: path.y, type: 'scatter', mode: 'lines', line: { color: '#f59e0b', width: 1.5 } },
            { x: [0], y: [0], type: 'scatter', mode: 'markers', marker: { color: '#ef4444', size: 8 } },
          ]}
          layout={mergeLayout({
            title: { text: 'Sample 2D Random Walk Path', font: { size: 13 } },
            xaxis: { title: { text: 'x' } },
            yaxis: { title: { text: 'y' }, scaleanchor: 'x', scaleratio: 1 },
            showlegend: false,
            margin: { t: 40, r: 20, b: 50, l: 60 },
          })}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 320 }}
        />
      </div>
    </div>
  );
}
