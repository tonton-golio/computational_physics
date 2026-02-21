'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import type { SimulationComponentProps } from '@/shared/types/simulation';


/**
 * Build the G matrix for linear tomography.
 * Rays travel diagonally across an N x N grid.
 * Left earthquake: rays go from upper-left to seismograph on surface.
 * Right earthquake: rays go from upper-right to seismograph on surface.
 */
function makeG(N: number): number[][] {
  const N2 = N * N;
  const rows: number[][] = [];

  // G_left: flipped identity diagonals
  const gLeft: number[][] = [];
  for (let i = 0; i < N - 2; i++) {
    const mat: number[] = new Array(N2).fill(0);
    // np.flip(np.eye(N, k=-(1+i)), axis=0).flatten()
    for (let r = 0; r < N; r++) {
      const c = (N - 1 - r) - (1 + i);
      if (c >= 0 && c < N) {
        mat[r * N + c] = 1;
      }
    }
    gLeft.push(mat);
  }

  // G_right: identity diagonals
  const gRight: number[][] = [];
  for (let i = 0; i < N - 2; i++) {
    const mat: number[] = new Array(N2).fill(0);
    // np.eye(N, k=1+i).flatten()
    for (let r = 0; r < N; r++) {
      const c = r + (1 + i);
      if (c >= 0 && c < N) {
        mat[r * N + c] = 1;
      }
    }
    gRight.push(mat);
  }

  const z = new Array(N2).fill(0);

  // Build G = [z, G_left[::-1], z, z, G_right, z]
  rows.push([...z]);
  for (let i = gLeft.length - 1; i >= 0; i--) {
    rows.push([...gLeft[i]]);
  }
  rows.push([...z]);
  rows.push([...z]);
  for (let i = 0; i < gRight.length; i++) {
    rows.push([...gRight[i]]);
  }
  rows.push([...z]);

  // Scale by sqrt(2) * 1000
  const scale = Math.SQRT2 * 1000;
  for (const row of rows) {
    for (let j = 0; j < row.length; j++) {
      row[j] *= scale;
    }
  }

  return rows;
}

function makeM(N: number, top: number, bot: number, left: number, right: number): number[] {
  const m = new Array(N * N).fill(0);
  const anomaly = (1 / 5000) - (1 / 5200);
  for (let r = top; r < bot; r++) {
    for (let c = left; c < right; c++) {
      m[r * N + c] = anomaly;
    }
  }
  return m;
}

function matVecMul(G: number[][], v: number[]): number[] {
  return G.map(row => row.reduce((sum, g, j) => sum + g * v[j], 0));
}

function vecNorm(v: number[]): number {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}

/**
 * Solve Tikhonov regularized inverse: m = (G^T G + eps^2 I)^{-1} G^T d
 * Using a simplified approach via solving the normal equations with Gauss elimination.
 */
function solveTikhonov(G: number[][], dObs: number[], epsilon: number): number[] {
  const nParams = G[0].length;
  const nData = G.length;

  // Build G^T G
  const GtG: number[][] = [];
  for (let i = 0; i < nParams; i++) {
    GtG.push(new Array(nParams).fill(0));
  }
  for (let i = 0; i < nParams; i++) {
    for (let j = 0; j <= i; j++) {
      let s = 0;
      for (let k = 0; k < nData; k++) {
        s += G[k][i] * G[k][j];
      }
      GtG[i][j] = s;
      GtG[j][i] = s;
    }
    GtG[i][i] += epsilon * epsilon;
  }

  // Build G^T d
  const Gtd: number[] = new Array(nParams).fill(0);
  for (let i = 0; i < nParams; i++) {
    for (let k = 0; k < nData; k++) {
      Gtd[i] += G[k][i] * dObs[k];
    }
  }

  // Solve (GtG) m = Gtd via Gaussian elimination with partial pivoting
  const A = GtG.map(row => [...row]);
  const b = [...Gtd];
  const n = nParams;

  for (let col = 0; col < n; col++) {
    // Pivot
    let maxVal = Math.abs(A[col][col]);
    let maxRow = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(A[r][col]) > maxVal) {
        maxVal = Math.abs(A[r][col]);
        maxRow = r;
      }
    }
    [A[col], A[maxRow]] = [A[maxRow], A[col]];
    [b[col], b[maxRow]] = [b[maxRow], b[col]];

    if (Math.abs(A[col][col]) < 1e-20) continue;

    for (let r = col + 1; r < n; r++) {
      const factor = A[r][col] / A[col][col];
      for (let j = col; j < n; j++) {
        A[r][j] -= factor * A[col][j];
      }
      b[r] -= factor * b[col];
    }
  }

  // Back substitution
  const m = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let j = i + 1; j < n; j++) {
      s -= A[i][j] * m[j];
    }
    m[i] = Math.abs(A[i][i]) > 1e-20 ? s / A[i][i] : 0;
  }

  return m;
}

export default function LinearTomography({ id }: SimulationComponentProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [N, setN] = useState(12);
  const [top, setTop] = useState(4);
  const [bot, setBot] = useState(8);
  const [left, setLeft] = useState(3);
  const [right, setRight] = useState(7);
  const [nEps, setNEps] = useState(10);

  const result = useMemo(() => {
    const clampedTop = Math.min(top, bot);
    const clampedBot = Math.max(top, bot);
    const clampedLeft = Math.min(left, right);
    const clampedRight = Math.max(left, right);

    const G = makeG(N);
    const mTrue = makeM(N, clampedTop, clampedBot, clampedLeft, clampedRight);

    // Forward: d_pure = G @ m
    const dPure = matVecMul(G, mTrue);

    // Add noise
    const noiseScale = 1 / 18;
    const dPureNorm = vecNorm(dPure);
    // Deterministic pseudo-noise
    const noise: number[] = dPure.map((_, i) => Math.sin(i * 2.1 + 3.7) * 0.5);
    const noiseNorm = vecNorm(noise);
    const scaledNoise = noise.map(n => n * (noiseScale * dPureNorm / (noiseNorm || 1)));
    const dObs = dPure.map((d, i) => d + scaledNoise[i]);
    const nNorm = vecNorm(scaledNoise);

    // Solve with Tikhonov over epsilon range
    const epsValues: number[] = [];
    for (let i = 0; i < nEps; i++) {
      const logEps = -3 + (7 * i) / (nEps - 1);
      epsValues.push(Math.pow(10, logEps));
    }

    const residuals: number[] = [];
    const solutions: number[][] = [];

    for (const eps of epsValues) {
      const mSol = solveTikhonov(G, dObs, eps);
      solutions.push(mSol);
      const dPred = matVecMul(G, mSol);
      const resid = dObs.map((d, i) => d - dPred[i]);
      const residNorm = vecNorm(resid);
      residuals.push(Math.abs(residNorm - nNorm));
    }

    // Find optimal epsilon
    let minIdx = 0;
    for (let i = 1; i < residuals.length; i++) {
      if (residuals[i] < residuals[minIdx]) minIdx = i;
    }
    const mOpt = solutions[minIdx];
    const epsOpt = epsValues[minIdx];

    // Reshape m arrays to 2D for heatmap
    const mTrueGrid: number[][] = [];
    const mOptGrid: number[][] = [];
    for (let r = 0; r < N; r++) {
      mTrueGrid.push(mTrue.slice(r * N, (r + 1) * N));
      mOptGrid.push(mOpt.slice(r * N, (r + 1) * N));
    }

    // Compute ray coverage for G visualization
    const gSum = new Array(N * N).fill(0);
    for (const row of G) {
      for (let j = 0; j < row.length; j++) {
        gSum[j] += Math.abs(row[j]);
      }
    }
    const gSumGrid: number[][] = [];
    for (let r = 0; r < N; r++) {
      gSumGrid.push(gSum.slice(r * N, (r + 1) * N));
    }

    // Seismograph data (split into left and right earthquakes)
    const halfLen = Math.floor(dObs.length / 2);
    const dLeft = dObs.slice(0, halfLen);
    const dRight = dObs.slice(halfLen);

    return {
      mTrueGrid,
      mOptGrid,
      gSumGrid,
      dLeft,
      dRight,
      epsValues,
      residuals,
      epsOpt,
      N,
    };
  }, [N, top, bot, left, right, nEps]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Linear Tomography</h3>
      <p className="text-[var(--text-muted)] text-sm mb-4">
        Reconstruct subsurface density anomalies from seismograph arrival times using Tikhonov regularization.
        Adjust the anomaly position and grid size. The method sweeps over regularization parameter epsilon
        to find the optimal reconstruction.
      </p>
      <div className="grid grid-cols-3 lg:grid-cols-6 gap-3 mb-4">
        <div>
          <label className="text-[var(--text-muted)] text-xs">Grid N: {N}</label>
          <Slider min={6} max={18} step={1} value={[N]} onValueChange={([v]) => setN(v)} />
        </div>
        <div>
          <label className="text-[var(--text-muted)] text-xs">Top: {top}</label>
          <Slider min={0} max={N - 1} step={1} value={[top]} onValueChange={([v]) => setTop(v)} />
        </div>
        <div>
          <label className="text-[var(--text-muted)] text-xs">Bottom: {bot}</label>
          <Slider min={1} max={N} step={1} value={[bot]} onValueChange={([v]) => setBot(v)} />
        </div>
        <div>
          <label className="text-[var(--text-muted)] text-xs">Left: {left}</label>
          <Slider min={0} max={N - 1} step={1} value={[left]} onValueChange={([v]) => setLeft(v)} />
        </div>
        <div>
          <label className="text-[var(--text-muted)] text-xs">Right: {right}</label>
          <Slider min={1} max={N} step={1} value={[right]} onValueChange={([v]) => setRight(v)} />
        </div>
        <div>
          <label className="text-[var(--text-muted)] text-xs">Eps steps: {nEps}</label>
          <Slider min={5} max={30} step={1} value={[nEps]} onValueChange={([v]) => setNEps(v)} />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
        <CanvasHeatmap
          data={[{
            z: result.mTrueGrid,
            type: 'heatmap',
            colorscale: 'Viridis',
          }]}
          layout={{
            title: { text: 'True Model m' },
            xaxis: { title: { text: 'x' } },
            yaxis: { title: { text: 'depth' }, autorange: 'reversed' as const },
            height: 300,
            margin: { t: 40, b: 50, l: 50, r: 20 },
          }}
          style={{ width: '100%' }}
        />
        <CanvasHeatmap
          data={[{
            z: result.gSumGrid,
            type: 'heatmap',
            colorscale: 'Greys',
          }]}
          layout={{
            title: { text: 'Ray Coverage (sum of |G|)' },
            xaxis: { title: { text: 'x' } },
            yaxis: { title: { text: 'depth' }, autorange: 'reversed' as const },
            height: 300,
            margin: { t: 40, b: 50, l: 50, r: 20 },
          }}
          style={{ width: '100%' }}
        />
        <CanvasHeatmap
          data={[{
            z: result.mOptGrid,
            type: 'heatmap',
            colorscale: 'Viridis',
          }]}
          layout={{
            title: { text: `Predicted m (eps=${result.epsOpt.toExponential(2)})` },
            xaxis: { title: { text: 'x' } },
            yaxis: { title: { text: 'depth' }, autorange: 'reversed' as const },
            height: 300,
            margin: { t: 40, b: 50, l: 50, r: 20 },
          }}
          style={{ width: '100%' }}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <CanvasChart
          data={[
            {
              x: Array.from({ length: result.dLeft.length }, (_, i) => i),
              y: result.dLeft,
              type: 'bar' as const,
              name: 'Left earthquake',
              marker: { color: 'rgba(59,130,246,0.7)' },
            },
            {
              x: Array.from({ length: result.dRight.length }, (_, i) => i),
              y: result.dRight,
              type: 'bar' as const,
              name: 'Right earthquake',
              marker: { color: 'rgba(251,146,60,0.7)' },
            },
          ]}
          layout={{
            title: { text: 'Observed Seismograph Data' },
            xaxis: { title: { text: 'Detector index' } },
            yaxis: { title: { text: 'Time anomaly' } },
            barmode: 'group',
            height: 300,
            margin: { t: 40, b: 50, l: 50, r: 20 },
          }}
          style={{ width: '100%' }}
        />
        <CanvasChart
          data={[{
            x: result.epsValues,
            y: result.residuals,
            type: 'scatter' as const,
            mode: 'lines+markers' as const,
            line: { color: '#f472b6' },
            marker: { color: '#f472b6', size: 5 },
            name: 'Residual',
          }]}
          layout={{
            title: { text: 'Residual vs Epsilon' },
            xaxis: { title: { text: 'epsilon' }, type: 'log' },
            yaxis: { title: { text: '|d_obs - G*m| - |noise|' }, type: 'log' },
            height: 300,
            margin: { t: 40, b: 50, l: 50, r: 20 },
            shapes: [{
              type: 'line',
              x0: result.epsOpt, x1: result.epsOpt,
              y0: 0, y1: 1, yref: 'paper',
              line: { color: '#22d3ee', dash: 'dash', width: 2 },
            }],
          }}
          style={{ width: '100%' }}
        />
      </div>
      <p className="text-[var(--text-soft)] text-xs mt-3">
        The optimal epsilon minimizes the difference between the residual norm and the noise norm.
        Areas traced by rays from both earthquakes are recovered well; untouched areas remain uncertain.
      </p>
    </div>
  );
}
