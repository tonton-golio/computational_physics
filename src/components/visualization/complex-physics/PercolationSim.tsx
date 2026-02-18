'use client';

import React, { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { usePlotlyTheme } from '@/lib/plotly-theme';
import { Slider } from '@/components/ui/slider';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

function generateGrid(size: number, seed: number): number[][] {
  // Simple seeded RNG (mulberry32)
  let s = seed;
  function rand() {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  const grid: number[][] = [];
  for (let i = 0; i < size; i++) {
    const row: number[] = [];
    for (let j = 0; j < size; j++) {
      row.push(rand());
    }
    grid.push(row);
  }
  return grid;
}

function findClusters(grid: number[][], p: number): { clusterMap: number[][]; numClusters: number } {
  const size = grid.length;
  const openGrid = grid.map(row => row.map(v => v < p));
  const clusterMap: number[][] = Array.from({ length: size }, () => Array(size).fill(-1));
  let clusterId = 0;

  // DFS using explicit stack to avoid recursion limit
  function dfs(startI: number, startJ: number, id: number) {
    const stack: [number, number][] = [[startI, startJ]];
    clusterMap[startI][startJ] = id;
    while (stack.length > 0) {
      const [ci, cj] = stack.pop()!;
      for (const [di, dj] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
        const ni = ci + di;
        const nj = cj + dj;
        if (ni >= 0 && ni < size && nj >= 0 && nj < size && openGrid[ni][nj] && clusterMap[ni][nj] === -1) {
          clusterMap[ni][nj] = id;
          stack.push([ni, nj]);
        }
      }
    }
  }

  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      if (openGrid[i][j] && clusterMap[i][j] === -1) {
        dfs(i, j, clusterId);
        clusterId++;
      }
    }
  }

  return { clusterMap, numClusters: clusterId };
}

// Generate distinct colors using HSL
function clusterColorScale(numClusters: number): string[] {
  const colors: string[] = [];
  for (let i = 0; i < numClusters; i++) {
    const hue = (i * 137.508) % 360; // golden angle for good distribution
    colors.push(`hsl(${hue}, 70%, 55%)`);
  }
  return colors;
}

export function PercolationSim() {
  const [p, setP] = useState(0.45);
  const [size, setSize] = useState(40);
  const [seed, setSeed] = useState(42);
  const { mergeLayout } = usePlotlyTheme();

  const { displayGrid, numClusters } = useMemo(() => {
    const grid = generateGrid(size, seed);
    const { clusterMap, numClusters } = findClusters(grid, p);
    clusterColorScale(numClusters);

    // Build display grid: -1 for closed sites, cluster ID for open sites
    const displayGrid: number[][] = clusterMap.map(row => row.map(v => v));
    return { displayGrid, numClusters };
  }, [p, size, seed]);

  // Percolation over many p values
  const npData = useMemo(() => {
    const grid = generateGrid(size, seed);
    const ps: number[] = [];
    const ns: number[] = [];
    for (let pp = 0.01; pp <= 0.95; pp += 0.05) {
      ps.push(pp);
      const { numClusters: nc } = findClusters(grid, pp);
      ns.push(nc);
    }
    return { ps, ns };
  }, [size, seed]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-6 items-center">
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">p = {p.toFixed(2)}</label>
          <Slider
            min={0.01}
            max={1.0}
            step={0.01}
            value={[p]}
            onValueChange={([v]) => setP(v)}
            className="w-48"
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">Size: {size}</label>
          <Slider
            min={10}
            max={80}
            step={5}
            value={[size]}
            onValueChange={([v]) => setSize(v)}
            className="w-48"
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">Seed: {seed}</label>
          <Slider
            min={1}
            max={200}
            step={1}
            value={[seed]}
            onValueChange={([v]) => setSeed(v)}
            className="w-48"
          />
        </div>
      </div>

      <div className="text-sm text-[var(--text-muted)]">
        N(p={p.toFixed(2)}) = {numClusters} clusters
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Plot
          data={[{
            z: displayGrid,
            type: 'heatmap',
            colorscale: 'Portland',
            showscale: false,
          }]}
          layout={mergeLayout({
            title: { text: `Site Percolation (p=${p.toFixed(2)})`, font: { size: 13 } },
            xaxis: { showticklabels: false },
            yaxis: { showticklabels: false },
            margin: { t: 40, r: 5, b: 5, l: 5 },
          })}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 400 }}
        />
        <Plot
          data={[{
            x: npData.ps,
            y: npData.ns,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#3b82f6', width: 2 },
            marker: { size: 5 },
          }]}
          layout={mergeLayout({
            title: { text: 'Number of Clusters N(p)', font: { size: 13 } },
            xaxis: { title: { text: 'p' } },
            yaxis: { title: { text: 'N' } },
            margin: { t: 40, r: 20, b: 50, l: 60 },
          })}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 400 }}
        />
      </div>
    </div>
  );
}
