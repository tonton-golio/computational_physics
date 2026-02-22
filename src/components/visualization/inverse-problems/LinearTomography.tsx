"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * Build rays for tomography - each ray is a path through the grid.
 * Left source: rays travel diagonally down-right
 * Right source: rays travel diagonally down-left
 */
function buildRays(N: number): { path: [number, number][] }[] {
  const rays: { path: [number, number][] }[] = [];

  // Left source rays (top-left corner, radiating to bottom and right edges)
  for (let startRow = 0; startRow < N; startRow++) {
    const path: [number, number][] = [];
    let r = startRow;
    let c = 0;
    while (r < N && c < N) {
      path.push([r, c]);
      r++;
      c++;
    }
    if (path.length > 1) rays.push({ path });
  }

  // Right source rays (top-right corner, radiating to bottom and left edges)
  for (let startRow = 0; startRow < N; startRow++) {
    const path: [number, number][] = [];
    let r = startRow;
    let c = N - 1;
    while (r < N && c >= 0) {
      path.push([r, c]);
      r++;
      c--;
    }
    if (path.length > 1) rays.push({ path });
  }

  return rays;
}

function makeTrueModel(N: number, anomalyRow: number, anomalyCol: number): number[] {
  const m = new Array(N * N).fill(0);
  const anomalyVal = 0.5;

  // Create a centered anomaly
  for (let dr = -1; dr <= 1; dr++) {
    for (let dc = -1; dc <= 1; dc++) {
      const r = anomalyRow + dr;
      const c = anomalyCol + dc;
      if (r >= 0 && r < N && c >= 0 && c < N) {
        m[r * N + c] = anomalyVal;
      }
    }
  }
  return m;
}

function buildG(rays: { path: [number, number][] }[], N: number): number[][] {
  const N2 = N * N;
  return rays.map(ray => {
    const row = new Array(N2).fill(0);
    for (const [r, c] of ray.path) {
      row[r * N + c] = 1;
    }
    return row;
  });
}

function matVecMul(G: number[][], v: number[]): number[] {
  return G.map(row => row.reduce((sum, g, j) => sum + g * v[j], 0));
}

function vecNorm(v: number[]): number {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}

function solveTikhonov(G: number[][], d: number[], epsilon: number): number[] {
  const n = G[0].length;
  const m = G.length;

  // G^T G + eps^2 I
  const AtA: number[][] = [];
  for (let i = 0; i < n; i++) {
    AtA.push(new Array(n).fill(0));
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = 0;
      for (let k = 0; k < m; k++) s += G[k][i] * G[k][j];
      AtA[i][j] = s;
      AtA[j][i] = s;
    }
    AtA[i][i] += epsilon * epsilon;
  }

  // G^T d
  const Atd: number[] = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < m; k++) Atd[i] += G[k][i] * d[k];
  }

  // Gaussian elimination
  const A = AtA.map(row => [...row]);
  const b = [...Atd];
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(A[r][col]) > Math.abs(A[maxRow][col])) maxRow = r;
    }
    [A[col], A[maxRow]] = [A[maxRow], A[col]];
    [b[col], b[maxRow]] = [b[maxRow], b[col]];
    if (Math.abs(A[col][col]) < 1e-20) continue;
    for (let r = col + 1; r < n; r++) {
      const f = A[r][col] / A[col][col];
      for (let j = col; j < n; j++) A[r][j] -= f * A[col][j];
      b[r] -= f * b[col];
    }
  }

  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let j = i + 1; j < n; j++) s -= A[i][j] * x[j];
    x[i] = Math.abs(A[i][i]) > 1e-20 ? s / A[i][i] : 0;
  }
  return x;
}

// SVG component for ray visualization
function RayGrid({ N, rays, anomalyRow, anomalyCol }: { N: number; rays: { path: [number, number][] }[]; anomalyRow: number; anomalyCol: number }) {
  const cellSize = 20;
  const width = N * cellSize;
  const height = N * cellSize;

  return (
    <svg width={width} height={height} className="border border-[var(--border)] rounded">
      {/* Grid cells */}
      {Array.from({ length: N * N }).map((_, i) => {
        const r = Math.floor(i / N);
        const c = i % N;
        const isAnomaly = Math.abs(r - anomalyRow) <= 1 && Math.abs(c - anomalyCol) <= 1;
        return (
          <rect
            key={i}
            x={c * cellSize}
            y={r * cellSize}
            width={cellSize}
            height={cellSize}
            fill={isAnomaly ? 'rgba(239, 68, 68, 0.4)' : 'transparent'}
            stroke="var(--border)"
            strokeWidth={0.5}
          />
        );
      })}
      {/* Rays */}
      {rays.map((ray, idx) => {
        const isLeft = idx < rays.length / 2;
        const points = ray.path.map(([r, c]) =>
          `${c * cellSize + cellSize / 2},${r * cellSize + cellSize / 2}`
        ).join(' ');
        return (
          <polyline
            key={idx}
            points={points}
            fill="none"
            stroke={isLeft ? '#3b82f6' : '#fb923c'}
            strokeWidth={1.5}
            opacity={0.6}
          />
        );
      })}
      {/* Source markers */}
      <circle cx={cellSize / 2} cy={cellSize / 2} r={4} fill="#3b82f6" />
      <circle cx={width - cellSize / 2} cy={cellSize / 2} r={4} fill="#fb923c" />
      {/* Anomaly marker */}
      <circle
        cx={anomalyCol * cellSize + cellSize / 2}
        cy={anomalyRow * cellSize + cellSize / 2}
        r={6}
        fill="none"
        stroke="#ef4444"
        strokeWidth={2}
      />
    </svg>
  );
}

export default function LinearTomography({}: SimulationComponentProps) {
  const [N, setN] = useState(10);
  const [anomalyRow, setAnomalyRow] = useState(5);
  const [anomalyCol, setAnomalyCol] = useState(5);
  const [epsilon, setEpsilon] = useState(0.1);

  const result = useMemo(() => {
    const rays = buildRays(N);
    const mTrue = makeTrueModel(N, anomalyRow, anomalyCol);
    const G = buildG(rays, N);

    // Forward model with noise
    const dClean = matVecMul(G, mTrue);
    const noise = dClean.map((_, i) => Math.sin(i * 2.3) * 0.02);
    const dObs = dClean.map((d, i) => d + noise[i]);

    // Inversion
    const mPred = solveTikhonov(G, dObs, epsilon);

    // Reshape for heatmaps
    const mTrueGrid: number[][] = [];
    const mPredGrid: number[][] = [];
    for (let r = 0; r < N; r++) {
      mTrueGrid.push(mTrue.slice(r * N, (r + 1) * N));
      mPredGrid.push(mPred.slice(r * N, (r + 1) * N));
    }

    return { rays, mTrueGrid, mPredGrid, G, dObs };
  }, [N, anomalyRow, anomalyCol, epsilon]);

  return (
    <SimulationPanel
      title="Tomography: Seeing Inside"
      caption="Rays from two sources (blue/orange) cross the grid. Can you reconstruct the hidden anomaly (red) from ray travel times? Move the anomaly to see how ray coverage affects resolution."
    >
      <SimulationConfig>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <SimulationLabel className="text-[var(--text-muted)] text-xs">Grid Size: {N}</SimulationLabel>
            <Slider min={6} max={14} step={1} value={[N]} onValueChange={([v]) => setN(v)} />
          </div>
          <div>
            <SimulationLabel className="text-[var(--text-muted)] text-xs">Anomaly Row: {anomalyRow}</SimulationLabel>
            <Slider min={2} max={N - 3} step={1} value={[anomalyRow]} onValueChange={([v]) => setAnomalyRow(v)} />
          </div>
          <div>
            <SimulationLabel className="text-[var(--text-muted)] text-xs">Anomaly Col: {anomalyCol}</SimulationLabel>
            <Slider min={2} max={N - 3} step={1} value={[anomalyCol]} onValueChange={([v]) => setAnomalyCol(v)} />
          </div>
          <div>
            <SimulationLabel className="text-[var(--text-muted)] text-xs">Smoothing: {epsilon.toFixed(2)}</SimulationLabel>
            <Slider min={0.01} max={1} step={0.01} value={[epsilon]} onValueChange={([v]) => setEpsilon(v)} />
          </div>
        </div>
      </SimulationConfig>

      <SimulationMain>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Ray visualization */}
          <div className="flex flex-col items-center">
            <h4 className="text-sm font-medium mb-2 text-[var(--text)]">Ray Geometry</h4>
            <RayGrid N={N} rays={result.rays} anomalyRow={anomalyRow} anomalyCol={anomalyCol} />
            <p className="text-xs text-[var(--text-soft)] mt-2 text-center">
              Blue rays from top-left, orange from top-right
            </p>
          </div>

          {/* True model */}
          <div className="flex flex-col items-center">
            <h4 className="text-sm font-medium mb-2 text-[var(--text)]">True Model</h4>
            <CanvasHeatmap
              data={[{
                z: result.mTrueGrid,
                type: 'heatmap',
                colorscale: [[0, '#1e293b'], [1, '#ef4444']],
                showscale: false,
              }]}
              layout={{
                xaxis: { visible: false },
                yaxis: { visible: false, autorange: 'reversed' as const },
                margin: { t: 0, b: 0, l: 0, r: 0 },
              }}
              style={{ width: 200, height: 200 }}
            />
          </div>

          {/* Predicted model */}
          <div className="flex flex-col items-center">
            <h4 className="text-sm font-medium mb-2 text-[var(--text)]">Reconstructed</h4>
            <CanvasHeatmap
              data={[{
                z: result.mPredGrid,
                type: 'heatmap',
                colorscale: [[0, '#1e293b'], [1, '#22c55e']],
                showscale: false,
              }]}
              layout={{
                xaxis: { visible: false },
                yaxis: { visible: false, autorange: 'reversed' as const },
                margin: { t: 0, b: 0, l: 0, r: 0 },
              }}
              style={{ width: 200, height: 200 }}
            />
          </div>
        </div>

        <div className="mt-4 p-3 bg-[var(--surface-alt)] rounded-lg">
          <p className="text-xs text-[var(--text-soft)]">
            <strong className="text-[var(--text)]">Key insight:</strong> Areas crossed by many rays (center) are well-resolved.
            Edges with few rays remain uncertain. Move the anomaly to the corners to see resolution degrade.
          </p>
        </div>
      </SimulationMain>
    </SimulationPanel>
  );
}
