'use client';

import React, { useState, useMemo, useEffect, useRef } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';
import { useTheme } from '@/lib/use-theme';

// ---------------------------------------------------------------------------
// Simulation logic
// ---------------------------------------------------------------------------

function generateGrid(size: number, seed: number): number[][] {
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

// ---------------------------------------------------------------------------
// 2D Grid visualization
// ---------------------------------------------------------------------------

function GridCanvas({ clusterMap, size, isDark }: { clusterMap: number[][]; size: number; isDark: boolean }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const redraw = () => {
      const width = container.clientWidth;
      const height = width;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.scale(dpr, dpr);

      const cellSize = width / size;
      const gap = Math.max(cellSize * 0.06, 0.5);

      ctx.fillStyle = isDark ? '#0a0a0f' : '#f0f4ff';
      ctx.fillRect(0, 0, width, height);

      for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
          const clusterId = clusterMap[i][j];
          if (clusterId >= 0) {
            const hue = (clusterId * 137.508) % 360;
            ctx.fillStyle = isDark ? `hsl(${hue}, 70%, 55%)` : `hsl(${hue}, 65%, 50%)`;
          } else {
            ctx.fillStyle = isDark ? '#1a1a2e' : '#dfe8fb';
          }
          ctx.fillRect(
            j * cellSize + gap / 2,
            i * cellSize + gap / 2,
            cellSize - gap,
            cellSize - gap,
          );
        }
      }
    };

    redraw();
    const ro = new ResizeObserver(redraw);
    ro.observe(container);
    return () => ro.disconnect();
  }, [clusterMap, size, isDark]);

  return (
    <div
      ref={containerRef}
      className="w-full rounded-lg overflow-hidden"
      style={{ aspectRatio: '1 / 1', background: isDark ? '#0a0a0f' : '#f0f4ff' }}
    >
      <canvas ref={canvasRef} style={{ display: 'block' }} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main exported component
// ---------------------------------------------------------------------------

export function PercolationSim() {
  const theme = useTheme();
  const isDark = theme === 'dark';
  const [p, setP] = useState(0.5);
  const [size, setSize] = useState(30);
  const [seed, setSeed] = useState(42);

  const grid = useMemo(() => generateGrid(size, seed), [size, seed]);
  const { clusterMap, numClusters } = useMemo(() => findClusters(grid, p), [grid, p]);

  const npData = useMemo(() => {
    const ps: number[] = [];
    const ns: number[] = [];
    for (let pp = 0.01; pp <= 0.95; pp += 0.05) {
      ps.push(pp);
      const { numClusters: nc } = findClusters(grid, pp);
      ns.push(nc);
    }
    return { ps, ns };
  }, [grid]);

  return (
    <div className="space-y-6">
      {/* Controls */}
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
            max={60}
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
        {/* 2D Cluster Grid */}
        <GridCanvas clusterMap={clusterMap} size={size} isDark={isDark} />

        {/* N(p) Curve */}
        <CanvasChart
          data={[{
            x: npData.ps,
            y: npData.ns,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#3b82f6', width: 2 },
            marker: { size: 5 },
          }]}
          layout={{
            title: { text: 'Number of Clusters N(p)', font: { size: 13 } },
            xaxis: { title: { text: 'p' } },
            yaxis: { title: { text: 'N' } },
            margin: { t: 40, r: 20, b: 50, l: 60 },
          }}
          style={{ width: '100%', height: 360 }}
        />
      </div>
    </div>
  );
}
