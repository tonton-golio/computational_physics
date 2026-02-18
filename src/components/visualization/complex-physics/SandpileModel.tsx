'use client';

import React, { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';
import { Slider } from '@/components/ui/slider';
import { usePlotlyTheme } from '@/lib/plotly-theme';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SandpileResult {
  grid: number[][];
  avalanches: number[];
}

function mulberry32(seed: number) {
  let s = seed >>> 0;
  return () => {
    s += 0x6D2B79F5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function runSandpile(size: number, nsteps: number, runSeed: number): SandpileResult {
  const rand = mulberry32((size * 97 + nsteps * 3 + runSeed * 131) >>> 0);
  const grid = Array.from({ length: size }, () => Array(size).fill(0));
  const avalanches: number[] = [];

  function topple(i: number, j: number): number {
    let count = 0;
    const stack: [number, number][] = [[i, j]];
    while (stack.length > 0) {
      const [x, y] = stack.pop()!;
      while (grid[x][y] >= 4) {
        grid[x][y] -= 4;
        count++;
        if (x > 0) {
          grid[x - 1][y] += 1;
          if (grid[x - 1][y] >= 4) stack.push([x - 1, y]);
        }
        if (x < size - 1) {
          grid[x + 1][y] += 1;
          if (grid[x + 1][y] >= 4) stack.push([x + 1, y]);
        }
        if (y > 0) {
          grid[x][y - 1] += 1;
          if (grid[x][y - 1] >= 4) stack.push([x, y - 1]);
        }
        if (y < size - 1) {
          grid[x][y + 1] += 1;
          if (grid[x][y + 1] >= 4) stack.push([x, y + 1]);
        }
      }
    }
    return count;
  }

  for (let t = 0; t < nsteps; t++) {
    const i = Math.floor(rand() * size);
    const j = Math.floor(rand() * size);
    grid[i][j] += 1;
    if (grid[i][j] >= 4) {
      const a = topple(i, j);
      if (a > 0) avalanches.push(a);
    } else {
      avalanches.push(0);
    }
  }
  return { grid, avalanches };
}

export function SandpileModel() {
  const [size, setSize] = useState(35);
  const [nsteps, setNsteps] = useState(5000);
  const [rerun, setRerun] = useState(0);
  const { mergeLayout } = usePlotlyTheme();

  const { grid, avalanches } = useMemo(() => runSandpile(size, nsteps, rerun), [size, nsteps, rerun]);
  const nonZero = avalanches.filter(v => v > 0);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-6 items-center">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Grid Size: {size}</label>
          <Slider value={[size]} onValueChange={([v]) => setSize(v)} min={15} max={80} step={5} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Steps: {nsteps}</label>
          <Slider value={[nsteps]} onValueChange={([v]) => setNsteps(v)} min={500} max={20000} step={500} />
        </div>
        <button onClick={() => setRerun(v => v + 1)} className="px-4 py-2 bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white rounded text-sm mt-4">
          Re-run
        </button>
      </div>

      <div className="text-sm text-[var(--text-muted)]">
        Non-zero avalanches: {nonZero.length} / {avalanches.length}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Plot
          data={[{ z: grid, type: 'heatmap', colorscale: 'Turbo', showscale: true }]}
          layout={mergeLayout({
            title: { text: 'Sandpile Height Field', font: { size: 13 } },
            xaxis: { visible: false },
            yaxis: { visible: false },
            margin: { t: 40, r: 10, b: 10, l: 10 },
          })}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 360 }}
        />
        <Plot
          data={[
            {
              x: nonZero.length > 0 ? nonZero : [0],
              type: 'histogram',
              marker: { color: '#f59e0b' },
              nbinsx: 50,
            },
          ]}
          layout={mergeLayout({
            title: { text: 'Avalanche Size Distribution', font: { size: 13 } },
            xaxis: { title: { text: 'Avalanche size' }, type: 'log' },
            yaxis: { title: { text: 'Count' }, type: 'log' },
            margin: { t: 40, r: 20, b: 50, l: 60 },
          })}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 360 }}
        />
      </div>
    </div>
  );
}
