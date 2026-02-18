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

function countComponents(adj: number[][]): number {
  const n = adj.length;
  const seen = Array(n).fill(false);
  let comps = 0;
  for (let i = 0; i < n; i++) {
    if (seen[i]) continue;
    comps++;
    const stack = [i];
    seen[i] = true;
    while (stack.length > 0) {
      const node = stack.pop()!;
      for (let j = 0; j < n; j++) {
        if (adj[node][j] > 0 && !seen[j]) {
          seen[j] = true;
          stack.push(j);
        }
      }
    }
  }
  return comps;
}

function generateRandomRegularLike(n: number, degree: number, seed: number): number[][] {
  const rand = mulberry32(seed);
  const adj = Array.from({ length: n }, () => Array(n).fill(0));
  const nodes = Array.from({ length: n }, (_, i) => i);
  for (let k = 0; k < degree; k++) {
    for (let i = nodes.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [nodes[i], nodes[j]] = [nodes[j], nodes[i]];
    }
    for (let i = 0; i + 1 < n; i += 2) {
      const a = nodes[i];
      const b = nodes[i + 1];
      if (a !== b) {
        adj[a][b] = 1;
        adj[b][a] = 1;
      }
    }
  }
  return adj;
}

function applyPercolation(adj: number[][], p: number, seed: number): number[][] {
  const rand = mulberry32(seed);
  const n = adj.length;
  const out = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (adj[i][j] > 0 && rand() < p) {
        out[i][j] = 1;
        out[j][i] = 1;
      }
    }
  }
  return out;
}

export function BetheLatticePercolation() {
  const [size, setSize] = useState(36);
  const [degree, setDegree] = useState(3);
  const [p, setP] = useState(0.35);
  const [seed, setSeed] = useState(11);
  const { mergeLayout } = usePlotlyTheme();

  const { percolatedAdj, npCurve, theoreticalPc } = useMemo(() => {
    const adj = generateRandomRegularLike(size, degree, seed);
    const perc = applyPercolation(adj, p, seed + 17);
    const ps: number[] = [];
    const Ns: number[] = [];
    for (let pp = 0.05; pp <= 0.95; pp += 0.05) {
      ps.push(Number(pp.toFixed(2)));
      const g = applyPercolation(adj, pp, seed + 97);
      Ns.push(countComponents(g));
    }
    const pc = 1 / Math.max(degree - 1, 1);
    return { percolatedAdj: perc, npCurve: { ps, Ns }, theoreticalPc: pc };
  }, [size, degree, p, seed]);

  const nodeX = Array.from({ length: size }, (_, i) => Math.cos((2 * Math.PI * i) / size));
  const nodeY = Array.from({ length: size }, (_, i) => Math.sin((2 * Math.PI * i) / size));
  const edgeX: number[] = [];
  const edgeY: number[] = [];
  for (let i = 0; i < size; i++) {
    for (let j = i + 1; j < size; j++) {
      if (percolatedAdj[i][j] > 0) {
        edgeX.push(nodeX[i], nodeX[j], NaN);
        edgeY.push(nodeY[i], nodeY[j], NaN);
      }
    }
  }

  const connected = countComponents(percolatedAdj);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-6 items-center">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Nodes: {size}</label>
          <Slider value={[size]} onValueChange={([v]) => setSize(v)} min={12} max={90} step={2} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Degree: {degree}</label>
          <Slider value={[degree]} onValueChange={([v]) => setDegree(v)} min={2} max={6} step={1} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">p: {p.toFixed(2)}</label>
          <Slider value={[p]} onValueChange={([v]) => setP(v)} min={0.01} max={1} step={0.01} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Seed: {seed}</label>
          <Slider value={[seed]} onValueChange={([v]) => setSeed(v)} min={1} max={200} step={1} />
        </div>
      </div>

      <div className="text-sm text-[var(--text-muted)]">
        Theoretical Bethe threshold: p_c = 1/(z-1) = {theoreticalPc.toFixed(3)} | observed connected components N(p) = {connected}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Plot
          data={[
            { x: edgeX, y: edgeY, type: 'scatter', mode: 'lines', line: { color: '#60a5fa', width: 1 }, hoverinfo: 'skip' },
            { x: nodeX, y: nodeY, type: 'scatter', mode: 'markers', marker: { color: '#f59e0b', size: 6 }, hoverinfo: 'skip' },
          ]}
          layout={mergeLayout({
            title: { text: 'Percolated Bethe-like Graph', font: { size: 13 } },
            xaxis: { visible: false },
            yaxis: { visible: false, scaleanchor: 'x', scaleratio: 1 },
            margin: { t: 40, r: 10, b: 10, l: 10 },
            showlegend: false,
          })}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 360 }}
        />
        <Plot
          data={[
            { x: npCurve.ps, y: npCurve.Ns, type: 'scatter', mode: 'lines+markers', line: { color: '#34d399', width: 2 }, marker: { size: 5 } },
            { x: [theoreticalPc, theoreticalPc], y: [Math.min(...npCurve.Ns), Math.max(...npCurve.Ns)], type: 'scatter', mode: 'lines', line: { color: '#ef4444', dash: 'dash' } },
          ]}
          layout={mergeLayout({
            title: { text: 'N(p) for Bethe-like Percolation', font: { size: 13 } },
            xaxis: { title: { text: 'p' } },
            yaxis: { title: { text: 'Connected Components N' } },
            showlegend: false,
            margin: { t: 40, r: 20, b: 50, l: 60 },
          })}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 360 }}
        />
      </div>
    </div>
  );
}
