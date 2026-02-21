'use client';

import React, { useState, useMemo, useEffect, useRef } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationMain } from '@/components/ui/simulation-main';
import { Slider } from '@/components/ui/slider';
import { useTheme } from '@/lib/use-theme';

// ---------------------------------------------------------------------------
// Lattice types
// ---------------------------------------------------------------------------

type LatticeType = 'square' | 'triangular' | 'honeycomb';

const LATTICE_INFO: Record<LatticeType, { label: string; pc: number; pcLabel: string }> = {
  square:     { label: 'Square (z = 4)',     pc: 0.5927, pcLabel: 'p_c ≈ 0.593' },
  triangular: { label: 'Triangular (z = 6)', pc: 0.5,    pcLabel: 'p_c = 0.500' },
  honeycomb:  { label: 'Honeycomb (z = 3)',   pc: 0.6962, pcLabel: 'p_c ≈ 0.696' },
};

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

function getNeighbors(i: number, j: number, size: number, lattice: LatticeType): [number, number][] {
  const neighbors: [number, number][] = [];
  const add = (ni: number, nj: number) => {
    if (ni >= 0 && ni < size && nj >= 0 && nj < size) neighbors.push([ni, nj]);
  };

  switch (lattice) {
    case 'square':
      add(i - 1, j); add(i + 1, j); add(i, j - 1); add(i, j + 1);
      break;
    case 'triangular':
      // 6 neighbors: 4 cardinal + 2 diagonals (offset depends on row parity)
      add(i - 1, j); add(i + 1, j); add(i, j - 1); add(i, j + 1);
      if (i % 2 === 0) {
        add(i - 1, j - 1); add(i + 1, j - 1);
      } else {
        add(i - 1, j + 1); add(i + 1, j + 1);
      }
      break;
    case 'honeycomb':
      // 3 neighbors: left, right, and one vertical (alternating up/down)
      add(i, j - 1); add(i, j + 1);
      if ((i + j) % 2 === 0) {
        add(i - 1, j);
      } else {
        add(i + 1, j);
      }
      break;
  }
  return neighbors;
}

function findClusters(
  grid: number[][], p: number, lattice: LatticeType,
): { clusterMap: number[][]; numClusters: number } {
  const size = grid.length;
  const openGrid = grid.map(row => row.map(v => v < p));
  const clusterMap: number[][] = Array.from({ length: size }, () => Array(size).fill(-1));
  let clusterId = 0;

  function dfs(startI: number, startJ: number, id: number) {
    const stack: [number, number][] = [[startI, startJ]];
    clusterMap[startI][startJ] = id;
    while (stack.length > 0) {
      const [ci, cj] = stack.pop()!;
      for (const [ni, nj] of getNeighbors(ci, cj, size, lattice)) {
        if (openGrid[ni][nj] && clusterMap[ni][nj] === -1) {
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

function drawHexagon(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number) {
  ctx.beginPath();
  for (let k = 0; k < 6; k++) {
    const angle = (Math.PI / 3) * k - Math.PI / 6;
    const x = cx + r * Math.cos(angle);
    const y = cy + r * Math.sin(angle);
    if (k === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fill();
}

function cellColor(clusterId: number, isDark: boolean): string {
  if (clusterId >= 0) {
    const hue = (clusterId * 137.508) % 360;
    return isDark ? `hsl(${hue}, 70%, 55%)` : `hsl(${hue}, 65%, 50%)`;
  }
  return isDark ? '#1a1a2e' : '#dfe8fb';
}

function GridCanvas({
  clusterMap, size, isDark, lattice,
}: {
  clusterMap: number[][]; size: number; isDark: boolean; lattice: LatticeType;
}) {
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

      const bg = isDark ? '#0a0a0f' : '#f0f4ff';
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, width, height);

      if (lattice === 'square') {
        const cellW = width / size;
        const cellH = height / size;
        const gap = Math.max(cellW * 0.06, 0.5);
        for (let i = 0; i < size; i++) {
          for (let j = 0; j < size; j++) {
            ctx.fillStyle = cellColor(clusterMap[i][j], isDark);
            ctx.fillRect(
              j * cellW + gap / 2,
              i * cellH + gap / 2,
              cellW - gap,
              cellH - gap,
            );
          }
        }
      } else {
        // Offset layout for triangular / honeycomb
        const cellW = width / (size + 0.5);
        const cellH = height / size;
        const r = Math.min(cellW, cellH) * 0.44;

        for (let i = 0; i < size; i++) {
          const offset = (i % 2 === 1) ? cellW * 0.5 : 0;
          for (let j = 0; j < size; j++) {
            ctx.fillStyle = cellColor(clusterMap[i][j], isDark);
            const cx = j * cellW + cellW / 2 + offset;
            const cy = i * cellH + cellH / 2;

            if (lattice === 'triangular') {
              ctx.beginPath();
              ctx.arc(cx, cy, r, 0, Math.PI * 2);
              ctx.fill();
            } else {
              drawHexagon(ctx, cx, cy, r);
            }
          }
        }
      }
    };

    redraw();
    const ro = new ResizeObserver(redraw);
    ro.observe(container);
    return () => ro.disconnect();
  }, [clusterMap, size, isDark, lattice]);

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
  const [lattice, setLattice] = useState<LatticeType>('square');

  const grid = useMemo(() => generateGrid(size, seed), [size, seed]);
  const { clusterMap, numClusters } = useMemo(
    () => findClusters(grid, p, lattice), [grid, p, lattice],
  );

  const info = LATTICE_INFO[lattice];

  const npData = useMemo(() => {
    const ps: number[] = [];
    const ns: number[] = [];
    for (let pp = 0.01; pp <= 0.95; pp += 0.05) {
      ps.push(pp);
      const { numClusters: nc } = findClusters(grid, pp, lattice);
      ns.push(nc);
    }
    return { ps, ns };
  }, [grid, lattice]);

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex flex-wrap gap-6 items-center">
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">Lattice</label>
          <select
            value={lattice}
            onChange={(e) => setLattice(e.target.value as LatticeType)}
            className="bg-[var(--surface-3)] text-[var(--text-strong)] p-2 rounded border border-[var(--border-strong)] text-sm"
          >
            {(Object.keys(LATTICE_INFO) as LatticeType[]).map((key) => (
              <option key={key} value={key}>{LATTICE_INFO[key].label}</option>
            ))}
          </select>
        </div>
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
        N(p={p.toFixed(2)}) = {numClusters} clusters · {info.pcLabel}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* 2D Cluster Grid */}
        <SimulationMain scaleMode="contain">
          <GridCanvas clusterMap={clusterMap} size={size} isDark={isDark} lattice={lattice} />
        </SimulationMain>

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
            shapes: [{
              type: 'line',
              x0: info.pc,
              x1: info.pc,
              y0: 0,
              y1: 1,
              yref: 'paper',
              line: { color: '#ef4444', width: 1.5, dash: 'dash' },
            }],
          }}
          style={{ width: '100%', height: 360 }}
        />
      </div>
    </div>
  );
}
