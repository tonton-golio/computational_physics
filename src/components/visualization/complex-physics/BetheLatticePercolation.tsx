'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationMain } from '@/components/ui/simulation-main';
import { Slider } from '@/components/ui/slider';
import { useTheme } from '@/lib/use-theme';

// ---------------------------------------------------------------------------
// PRNG
// ---------------------------------------------------------------------------

function mulberry32(seed: number) {
  let s = seed >>> 0;
  return () => {
    s += 0x6D2B79F5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// Cayley tree (true Bethe lattice) generation
// ---------------------------------------------------------------------------

interface TreeNode {
  id: number;
  children: number[];
  parent: number | null;
  depth: number;
}

function generateCayleyTree(z: number, generations: number): { nodes: TreeNode[]; adj: number[][] } {
  const nodes: TreeNode[] = [];
  let nextId = 0;

  const root: TreeNode = { id: nextId++, children: [], parent: null, depth: 0 };
  nodes.push(root);

  let currentLevel = [root];
  for (let gen = 1; gen <= generations; gen++) {
    const nextLevel: TreeNode[] = [];
    for (const parent of currentLevel) {
      const numChildren = parent.parent === null ? z : z - 1;
      for (let c = 0; c < numChildren; c++) {
        const child: TreeNode = { id: nextId++, children: [], parent: parent.id, depth: gen };
        parent.children.push(child.id);
        nodes.push(child);
        nextLevel.push(child);
      }
    }
    currentLevel = nextLevel;
  }

  const n = nodes.length;
  const adj = Array.from({ length: n }, () => Array(n).fill(0));
  for (const node of nodes) {
    for (const childId of node.children) {
      adj[node.id][childId] = 1;
      adj[childId][node.id] = 1;
    }
  }

  return { nodes, adj };
}

function maxGenerationsForZ(z: number, maxNodes = 200): number {
  let total = 1;
  let levelSize = z;
  let gen = 0;
  while (total + levelSize <= maxNodes) {
    total += levelSize;
    levelSize *= Math.max(z - 1, 1);
    gen++;
    if (z === 2 && gen >= 12) break;
  }
  return Math.max(gen, 1);
}

// ---------------------------------------------------------------------------
// Percolation
// ---------------------------------------------------------------------------

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

function getComponents(adj: number[][]): { count: number; componentOf: number[] } {
  const n = adj.length;
  const componentOf = Array(n).fill(-1);
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (componentOf[i] >= 0) continue;
    const stack = [i];
    componentOf[i] = count;
    while (stack.length > 0) {
      const node = stack.pop()!;
      for (let j = 0; j < n; j++) {
        if (adj[node][j] > 0 && componentOf[j] < 0) {
          componentOf[j] = count;
          stack.push(j);
        }
      }
    }
    count++;
  }
  return { count, componentOf };
}

// ---------------------------------------------------------------------------
// Radial tree layout
// ---------------------------------------------------------------------------

function computeRadialLayout(nodes: TreeNode[], canvasSize: number): [number, number][] {
  const positions: [number, number][] = new Array(nodes.length);
  const cx = canvasSize / 2;
  const cy = canvasSize / 2;

  if (nodes.length <= 1) {
    positions[0] = [cx, cy];
    return positions;
  }

  const maxDepth = Math.max(...nodes.map(n => n.depth));
  const radiusStep = (canvasSize * 0.42) / Math.max(maxDepth, 1);

  // Pre-compute leaf counts bottom-up
  const leafCount = new Array(nodes.length).fill(0);
  for (let i = nodes.length - 1; i >= 0; i--) {
    if (nodes[i].children.length === 0) {
      leafCount[i] = 1;
    } else {
      leafCount[i] = nodes[i].children.reduce((sum, cid) => sum + leafCount[cid], 0);
    }
  }

  function layoutSubtree(nodeId: number, angleStart: number, angleEnd: number) {
    const node = nodes[nodeId];
    const radius = node.depth * radiusStep;
    const angle = (angleStart + angleEnd) / 2;

    if (node.depth === 0) {
      positions[nodeId] = [cx, cy];
    } else {
      positions[nodeId] = [
        cx + radius * Math.cos(angle),
        cy + radius * Math.sin(angle),
      ];
    }

    if (node.children.length === 0) return;

    const totalLeaves = leafCount[nodeId];
    let currentAngle = angleStart;
    for (const childId of node.children) {
      const span = (angleEnd - angleStart) * (leafCount[childId] / totalLeaves);
      layoutSubtree(childId, currentAngle, currentAngle + span);
      currentAngle += span;
    }
  }

  layoutSubtree(0, 0, 2 * Math.PI);
  return positions;
}

// ---------------------------------------------------------------------------
// Canvas tree visualization
// ---------------------------------------------------------------------------

function TreeCanvas({
  nodes,
  adj,
  percolatedAdj,
  componentOf,
  numComponents,
  isDark,
}: {
  nodes: TreeNode[];
  adj: number[][];
  percolatedAdj: number[][];
  componentOf: number[];
  numComponents: number;
  isDark: boolean;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const redraw = () => {
      const size = container.clientWidth;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = size * dpr;
      canvas.height = size * dpr;
      canvas.style.width = `${size}px`;
      canvas.style.height = `${size}px`;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.scale(dpr, dpr);

      ctx.fillStyle = isDark ? '#0a0a0f' : '#f0f4ff';
      ctx.fillRect(0, 0, size, size);

      const n = nodes.length;
      const positions = computeRadialLayout(nodes, size);
      const nodeRadius = Math.max(2, Math.min(6, 120 / Math.sqrt(n)));

      // Original tree edges (faded)
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          if (adj[i][j] > 0) {
            const [x1, y1] = positions[i];
            const [x2, y2] = positions[j];
            ctx.strokeStyle = isDark ? 'rgba(60, 80, 120, 0.25)' : 'rgba(60, 80, 120, 0.2)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }
        }
      }

      // Percolated edges (bright with glow)
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          if (percolatedAdj[i][j] > 0) {
            const [x1, y1] = positions[i];
            const [x2, y2] = positions[j];
            ctx.strokeStyle = isDark ? 'rgba(68, 136, 255, 0.3)' : 'rgba(36, 88, 201, 0.25)';
            ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
            ctx.strokeStyle = isDark ? 'rgba(68, 136, 255, 0.9)' : 'rgba(36, 88, 201, 0.8)';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }
        }
      }

      // Nodes colored by connected component
      for (let i = 0; i < n; i++) {
        const [nx, ny] = positions[i];
        const compId = componentOf[i];
        const hue = numComponents <= 1 ? 35 : (compId * 137.508) % 360;

        if (isDark) {
          // Glow halo
          ctx.fillStyle = `hsla(${hue}, 80%, 55%, 0.15)`;
          ctx.beginPath();
          ctx.arc(nx, ny, nodeRadius * 2.5, 0, Math.PI * 2);
          ctx.fill();
        }

        // Outer ring
        ctx.fillStyle = isDark ? `hsla(${hue}, 80%, 55%, 0.6)` : `hsla(${hue}, 70%, 45%, 0.4)`;
        ctx.beginPath();
        ctx.arc(nx, ny, nodeRadius * 1.5, 0, Math.PI * 2);
        ctx.fill();

        // Core
        ctx.fillStyle = isDark ? `hsl(${hue}, 80%, 65%)` : `hsl(${hue}, 70%, 45%)`;
        ctx.beginPath();
        ctx.arc(nx, ny, nodeRadius, 0, Math.PI * 2);
        ctx.fill();

        // Highlight dot
        ctx.fillStyle = isDark ? `hsla(${hue}, 60%, 85%, 0.9)` : `hsla(${hue}, 50%, 75%, 0.9)`;
        ctx.beginPath();
        ctx.arc(nx, ny, nodeRadius * 0.4, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    redraw();
    const ro = new ResizeObserver(redraw);
    ro.observe(container);
    return () => ro.disconnect();
  }, [nodes, adj, percolatedAdj, componentOf, numComponents, isDark]);

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
// Main component
// ---------------------------------------------------------------------------

export function BetheLatticePercolation() {
  const theme = useTheme();
  const isDark = theme === 'dark';
  const [generations, setGenerations] = useState(4);
  const [degree, setDegree] = useState(3);
  const [p, setP] = useState(0.5);
  const [seed, setSeed] = useState(11);

  const maxGens = useMemo(() => maxGenerationsForZ(degree), [degree]);
  const effectiveGens = Math.min(generations, maxGens);

  const { nodes, adj, percolatedAdj, componentOf, numComponents, npCurve, theoreticalPc } = useMemo(() => {
    const { nodes, adj } = generateCayleyTree(degree, effectiveGens);
    const perc = applyPercolation(adj, p, seed + 17);
    const { count, componentOf } = getComponents(perc);

    const ps: number[] = [];
    const Ns: number[] = [];
    for (let pp = 0.05; pp <= 0.95; pp += 0.05) {
      ps.push(Number(pp.toFixed(2)));
      const g = applyPercolation(adj, pp, seed + 97);
      const { count: c } = getComponents(g);
      Ns.push(c);
    }

    const pc = 1 / Math.max(degree - 1, 1);
    return { nodes, adj, percolatedAdj: perc, componentOf, numComponents: count, npCurve: { ps, Ns }, theoreticalPc: pc };
  }, [degree, effectiveGens, p, seed]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-6 items-center">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Generations: {effectiveGens}</label>
          <Slider value={[effectiveGens]} onValueChange={([v]) => setGenerations(v)} min={1} max={maxGens} step={1} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Degree z: {degree}</label>
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
        {nodes.length} nodes | Theoretical Bethe threshold: p_c = 1/(z−1) = {theoreticalPc.toFixed(3)} | Connected components N(p) = {numComponents}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <SimulationMain scaleMode="contain">
          <TreeCanvas
            nodes={nodes}
            adj={adj}
            percolatedAdj={percolatedAdj}
            componentOf={componentOf}
            numComponents={numComponents}
            isDark={isDark}
          />
        </SimulationMain>

        <CanvasChart
          data={[
            {
              x: npCurve.ps,
              y: npCurve.Ns,
              type: 'scatter',
              mode: 'lines+markers',
              line: { color: '#34d399', width: 2 },
              marker: { size: 5 },
            },
            {
              x: [theoreticalPc, theoreticalPc],
              y: [Math.min(...npCurve.Ns), Math.max(...npCurve.Ns)],
              type: 'scatter',
              mode: 'lines',
              line: { color: '#ef4444', dash: 'dash' },
            },
          ]}
          layout={{
            title: { text: 'N(p) for Bethe Lattice Percolation', font: { size: 13 } },
            xaxis: { title: { text: 'p' } },
            yaxis: { title: { text: 'Connected Components N' } },
            showlegend: false,
            margin: { t: 40, r: 20, b: 50, l: 60 },
          }}
          style={{ width: '100%', height: 360 }}
        />
      </div>
    </div>
  );
}
