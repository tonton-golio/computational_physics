'use client';

import React, { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id: string;
}

const G_CONST = 6.674e-11;

/**
 * Compute the G matrix for gravity gradient inversion.
 * G_{j,i} = G_const * log( (z_base_i^2 + x_j^2) / (z_top_i^2 + x_j^2) )
 */
function buildGMatrix(xs: number[], zs: number[]): number[][] {
  const nX = xs.length;
  const nZ = zs.length;
  const G: number[][] = [];
  for (let j = 0; j < nX; j++) {
    const row: number[] = [];
    for (let i = 0; i < nZ; i++) {
      const zTop = i === 0 ? 0 : zs[i - 1];
      const zBase = zs[i];
      const xj2 = xs[j] * xs[j];
      const val = G_CONST * Math.log((zBase * zBase + xj2) / (zTop * zTop + xj2 + 1e-30));
      row.push(val);
    }
    G.push(row);
  }
  return G;
}

function matVecMul(G: number[][], v: number[]): number[] {
  return G.map(row => row.reduce((s, g, j) => s + g * v[j], 0));
}

function vecNorm(v: number[]): number {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}

/**
 * Tikhonov solve: m = (G^T G + eps^2 I)^{-1} G^T d
 */
function solveTikhonov(G: number[][], d: number[], eps: number): number[] {
  const nParams = G[0].length;
  const nData = G.length;

  // G^T G + eps^2 I
  const A: number[][] = [];
  for (let i = 0; i < nParams; i++) {
    A.push(new Array(nParams).fill(0));
  }
  for (let i = 0; i < nParams; i++) {
    for (let j = 0; j <= i; j++) {
      let s = 0;
      for (let k = 0; k < nData; k++) {
        s += G[k][i] * G[k][j];
      }
      A[i][j] = s;
      A[j][i] = s;
    }
    A[i][i] += eps * eps;
  }

  // G^T d
  const b: number[] = new Array(nParams).fill(0);
  for (let i = 0; i < nParams; i++) {
    for (let k = 0; k < nData; k++) {
      b[i] += G[k][i] * d[k];
    }
  }

  // Gaussian elimination
  const n = nParams;
  const M = A.map(row => [...row]);
  const rhs = [...b];

  for (let col = 0; col < n; col++) {
    let maxVal = Math.abs(M[col][col]);
    let maxRow = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > maxVal) {
        maxVal = Math.abs(M[r][col]);
        maxRow = r;
      }
    }
    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    [rhs[col], rhs[maxRow]] = [rhs[maxRow], rhs[col]];

    if (Math.abs(M[col][col]) < 1e-30) continue;

    for (let r = col + 1; r < n; r++) {
      const factor = M[r][col] / M[col][col];
      for (let j = col; j < n; j++) {
        M[r][j] -= factor * M[col][j];
      }
      rhs[r] -= factor * rhs[col];
    }
  }

  const m = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = rhs[i];
    for (let j = i + 1; j < n; j++) {
      s -= M[i][j] * m[j];
    }
    m[i] = Math.abs(M[i][i]) > 1e-30 ? s / M[i][i] : 0;
  }

  return m;
}

export default function TikhonovRegularization({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [nDepthCells, setNDepthCells] = useState(20);
  const [epsExpMin, setEpsExpMin] = useState(-13);
  const [epsExpMax, setEpsExpMax] = useState(-9);
  const [nEps, setNEps] = useState(16);

  const result = useMemo(() => {
    // Measurement positions (similar to the old gravdata)
    const nMeasurements = 18;
    const xs: number[] = [];
    for (let i = 0; i < nMeasurements; i++) {
      xs.push(-500 + i * 200);
    }

    // Depth cells
    const zs: number[] = [];
    for (let i = 0; i < nDepthCells; i++) {
      zs.push((i + 1) * 100);
    }

    const G = buildGMatrix(xs, zs);

    // Synthetic observed data: a density anomaly in the middle
    const mTrue: number[] = new Array(nDepthCells).fill(0);
    const anomalyStart = Math.floor(nDepthCells * 0.3);
    const anomalyEnd = Math.floor(nDepthCells * 0.6);
    for (let i = anomalyStart; i < anomalyEnd; i++) {
      mTrue[i] = 500; // density anomaly in kg/m^3
    }

    const dObs = matVecMul(G, mTrue);
    // Add small noise
    const dObsNoisy = dObs.map((d, i) => d + d * 0.02 * Math.sin(i * 3.14));

    // Sweep epsilon
    const epsValues: number[] = [];
    for (let i = 0; i < nEps; i++) {
      const logE = epsExpMin + (epsExpMax - epsExpMin) * i / (nEps - 1);
      epsValues.push(Math.pow(10, logE));
    }

    const solutions: number[][] = [];
    const residuals: number[] = [];
    const modelNorms: number[] = [];

    for (const eps of epsValues) {
      const mSol = solveTikhonov(G, dObsNoisy, eps);
      solutions.push(mSol);
      const dPred = matVecMul(G, mSol);
      const resid = dObsNoisy.map((d, i) => d - dPred[i]);
      residuals.push(vecNorm(resid));
      modelNorms.push(vecNorm(mSol));
    }

    // Build a 2D array for the heatmap: rows = epsilon, cols = depth
    const solutionGrid: number[][] = solutions;

    return {
      xs,
      zs,
      dObs: dObsNoisy,
      mTrue,
      epsValues,
      solutionGrid,
      residuals,
      modelNorms,
      nDepthCells,
    };
  }, [nDepthCells, epsExpMin, epsExpMax, nEps]);

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">Tikhonov Regularization: Epsilon Sweep</h3>
      <p className="text-gray-400 text-sm mb-4">
        Visualize how the regularization parameter epsilon affects the recovered density profile.
        Small epsilon yields noisy solutions (overfitting); large epsilon yields overly smooth solutions (underfitting).
      </p>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
        <div>
          <label className="text-gray-300 text-xs">Depth cells: {nDepthCells}</label>
          <input type="range" min={8} max={40} step={1} value={nDepthCells}
            onChange={(e) => setNDepthCells(parseInt(e.target.value))} className="w-full" />
        </div>
        <div>
          <label className="text-gray-300 text-xs">log10(eps_min): {epsExpMin}</label>
          <input type="range" min={-15} max={-5} step={1} value={epsExpMin}
            onChange={(e) => setEpsExpMin(parseInt(e.target.value))} className="w-full" />
        </div>
        <div>
          <label className="text-gray-300 text-xs">log10(eps_max): {epsExpMax}</label>
          <input type="range" min={-10} max={0} step={1} value={epsExpMax}
            onChange={(e) => setEpsExpMax(parseInt(e.target.value))} className="w-full" />
        </div>
        <div>
          <label className="text-gray-300 text-xs">Epsilon steps: {nEps}</label>
          <input type="range" min={5} max={30} step={1} value={nEps}
            onChange={(e) => setNEps(parseInt(e.target.value))} className="w-full" />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        <Plot
          data={[{
            z: result.solutionGrid,
            x: Array.from({ length: result.nDepthCells }, (_, i) => i),
            y: result.epsValues.map(e => e.toExponential(1)),
            type: 'heatmap' as const,
            colorscale: 'Inferno',
            colorbar: { title: { text: 'density' }, tickfont: { color: '#9ca3af' } },
          }]}
          layout={{
            title: { text: 'Solutions m(depth) for each epsilon' },
            xaxis: { title: { text: 'Depth index' }, color: '#9ca3af' },
            yaxis: { title: { text: 'epsilon' }, color: '#9ca3af' },
            height: 380,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(15,15,25,1)',
            font: { color: '#9ca3af' },
            margin: { t: 40, b: 50, l: 80, r: 20 },
          }}
          config={{ displayModeBar: false }}
          style={{ width: '100%' }}
        />
        <Plot
          data={[
            {
              x: result.residuals,
              y: result.modelNorms,
              text: result.epsValues.map(e => `eps=${e.toExponential(2)}`),
              type: 'scatter' as const,
              mode: 'text+lines+markers' as const,
              textposition: 'top right',
              textfont: { size: 8, color: '#9ca3af' },
              marker: { color: result.epsValues.map((_, i) => i), colorscale: 'Viridis', size: 8 },
              line: { color: 'rgba(255,255,255,0.3)' },
              name: 'L-curve',
            },
          ]}
          layout={{
            title: { text: 'L-Curve (Residual vs Model Norm)' },
            xaxis: { title: { text: '||d_obs - Gm||' }, type: 'log', color: '#9ca3af' },
            yaxis: { title: { text: '||m||' }, type: 'log', color: '#9ca3af' },
            height: 380,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(15,15,25,1)',
            font: { color: '#9ca3af' },
            margin: { t: 40, b: 50, l: 60, r: 20 },
          }}
          config={{ displayModeBar: false }}
          style={{ width: '100%' }}
        />
      </div>

      <Plot
        data={[
          {
            x: Array.from({ length: result.nDepthCells }, (_, i) => i),
            y: result.mTrue,
            type: 'scatter' as const,
            mode: 'lines' as const,
            line: { color: '#22d3ee', width: 3, dash: 'dash' },
            name: 'True model',
          },
          ...result.solutionGrid.filter((_, i) => i % Math.max(1, Math.floor(result.solutionGrid.length / 5)) === 0).map((sol, idx) => ({
            x: Array.from({ length: result.nDepthCells }, (_, i) => i),
            y: sol,
            type: 'scatter' as const,
            mode: 'lines' as const,
            line: { width: 1.5 },
            name: `eps=${result.epsValues[idx * Math.max(1, Math.floor(result.solutionGrid.length / 5))].toExponential(1)}`,
            opacity: 0.7,
          })),
        ]}
        layout={{
          title: { text: 'Selected Solutions vs True Model' },
          xaxis: { title: { text: 'Depth index' }, color: '#9ca3af' },
          yaxis: { title: { text: 'Density anomaly' }, color: '#9ca3af' },
          height: 350,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(15,15,25,1)',
          font: { color: '#9ca3af' },
          legend: { bgcolor: 'rgba(0,0,0,0.3)', x: 0.7, y: 0.98 },
          margin: { t: 40, b: 50, l: 60, r: 20 },
        }}
        config={{ displayModeBar: false }}
        style={{ width: '100%' }}
      />
      <p className="text-gray-500 text-xs mt-3">
        The L-curve shows the trade-off between data misfit and model complexity. The corner of the L-curve indicates the optimal regularization parameter.
        Precision increases as resolution is refined, but too little regularization amplifies noise.
      </p>
    </div>
  );
}
