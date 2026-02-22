"use client";

import { useState, useMemo } from 'react';
import { gaussianPair } from '@/lib/math';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => { s = (s * 16807 + 0) % 2147483647; return s / 2147483647; };
}
function vecNorm(v: number[]): number {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}
function solveTikhonov(GtG: number[][], Gtd: number[], eps: number, n: number): number[] {
  const A = GtG.map(row => [...row]), b = [...Gtd];
  for (let i = 0; i < n; i++) A[i][i] += eps * eps;
  for (let col = 0; col < n; col++) {
    let mx = Math.abs(A[col][col]), mr = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(A[r][col]) > mx) { mx = Math.abs(A[r][col]); mr = r; }
    [A[col], A[mr]] = [A[mr], A[col]]; [b[col], b[mr]] = [b[mr], b[col]];
    if (Math.abs(A[col][col]) < 1e-30) continue;
    for (let r = col + 1; r < n; r++) {
      const f = A[r][col] / A[col][col];
      for (let j = col; j < n; j++) A[r][j] -= f * A[col][j];
      b[r] -= f * b[col];
    }
  }
  const m = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let j = i + 1; j < n; j++) s -= A[i][j] * m[j];
    m[i] = Math.abs(A[i][i]) > 1e-30 ? s / A[i][i] : 0;
  }
  return m;
}

export default function LCurveConstruction({}: SimulationComponentProps) {
  const [noiseLevel, setNoiseLevel] = useState(0.05);
  const [nParams, setNParams] = useState(20);

  const result = useMemo(() => {
    const rng = seededRandom(42), nData = 30, n = nParams;
    const G: number[][] = [];
    for (let i = 0; i < nData; i++) {
      const row: number[] = [], ti = (i + 1) / nData;
      for (let j = 0; j < n; j++) { const sj = (j + 0.5) / n; row.push(Math.exp(-3 * (ti - sj) ** 2)); }
      G.push(row);
    }
    const mTrue = Array.from({ length: n }, (_, j) => {
      const s = (j + 0.5) / n;
      return Math.sin(2 * Math.PI * s) * Math.exp(-2 * (s - 0.5) ** 2);
    });
    const dObs = G.map(row => row.reduce((s, g, j) => s + g * mTrue[j], 0) + noiseLevel * gaussianPair(rng)[0]);
    const GtG: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++)
      for (let j = 0; j <= i; j++) {
        let s = 0; for (let k = 0; k < nData; k++) s += G[k][i] * G[k][j];
        GtG[i][j] = s; GtG[j][i] = s;
      }
    const Gtd = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let k = 0; k < nData; k++) Gtd[i] += G[k][i] * dObs[k];
    const nEps = 40, epsVals: number[] = [], residNorms: number[] = [], solNorms: number[] = [];
    for (let i = 0; i < nEps; i++) epsVals.push(Math.pow(10, -4 + 4 * i / (nEps - 1)));
    for (const eps of epsVals) {
      const mSol = solveTikhonov(GtG, Gtd, eps, n);
      residNorms.push(vecNorm(dObs.map((d, i) => d - G[i].reduce((s, g, j) => s + g * mSol[j], 0))));
      solNorms.push(vecNorm(mSol));
    }
    const logR = residNorms.map(r => Math.log10(Math.max(r, 1e-20)));
    const logM = solNorms.map(m => Math.log10(Math.max(m, 1e-20)));
    let maxCurv = -Infinity, cornerIdx = 1;
    for (let i = 2; i < nEps - 2; i++) {
      const dr1 = logR[i] - logR[i - 1], dm1 = logM[i] - logM[i - 1];
      const dr2 = logR[i + 1] - logR[i], dm2 = logM[i + 1] - logM[i];
      const curv = Math.abs(dr1 * dm2 - dr2 * dm1) / Math.pow(dr1 ** 2 + dm1 ** 2, 1.5 + 1e-12);
      if (curv > maxCurv) { maxCurv = curv; cornerIdx = i; }
    }
    const xModel = Array.from({ length: n }, (_, j) => (j + 0.5) / n);
    return {
      residNorms, solNorms, epsVals, cornerIdx, mTrue, xModel,
      mCorner: solveTikhonov(GtG, Gtd, epsVals[cornerIdx], n),
      mUnder: solveTikhonov(GtG, Gtd, epsVals[1], n),
      mOver: solveTikhonov(GtG, Gtd, epsVals[nEps - 2], n),
    };
  }, [noiseLevel, nParams]);

  return (
    <SimulationPanel title="L-Curve Method">
      <SimulationConfig>
        <div className="grid grid-cols-2 gap-6">
          <div>
            <SimulationLabel className="mb-1 block text-sm text-[var(--text-muted)]">Noise level: {noiseLevel.toFixed(3)}</SimulationLabel>
            <Slider value={[noiseLevel]} onValueChange={([v]) => setNoiseLevel(v)} min={0.001} max={0.3} step={0.001} />
          </div>
          <div>
            <SimulationLabel className="mb-1 block text-sm text-[var(--text-muted)]">Parameters: {nParams}</SimulationLabel>
            <Slider value={[nParams]} onValueChange={([v]) => setNParams(v)} min={8} max={40} step={1} />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        <CanvasChart
          data={[
            { x: result.residNorms, y: result.solNorms, type: 'scatter', mode: 'lines+markers',
              line: { color: '#3b82f6', width: 2 }, marker: { size: 3, color: '#3b82f6' }, name: 'L-curve' },
            { x: [result.residNorms[result.cornerIdx]], y: [result.solNorms[result.cornerIdx]],
              type: 'scatter', mode: 'markers', marker: { size: 12, color: '#f59e0b' },
              name: `Corner (eps=${result.epsVals[result.cornerIdx].toExponential(1)})` },
          ]}
          layout={{
            title: { text: 'L-Curve: ||residual|| vs ||model||' },
            xaxis: { title: { text: '||d - Gm||' }, type: 'log' },
            yaxis: { title: { text: '||m||' }, type: 'log' },
            height: 380, margin: { t: 40, b: 50, l: 60, r: 20 },
          }}
          style={{ width: '100%' }}
        />
        <CanvasChart
          data={[
            { x: result.xModel, y: result.mTrue, type: 'scatter', mode: 'lines',
              line: { color: '#22d3ee', width: 2, dash: 'dash' }, name: 'True model' },
            { x: result.xModel, y: result.mUnder, type: 'scatter', mode: 'lines',
              line: { color: '#ef4444', width: 1.5 }, name: 'Under-regularized' },
            { x: result.xModel, y: result.mCorner, type: 'scatter', mode: 'lines',
              line: { color: '#f59e0b', width: 2.5 }, name: 'L-curve corner' },
            { x: result.xModel, y: result.mOver, type: 'scatter', mode: 'lines',
              line: { color: '#8b5cf6', width: 1.5 }, name: 'Over-regularized' },
          ]}
          layout={{
            title: { text: 'Recovered Models' }, xaxis: { title: { text: 's' } },
            yaxis: { title: { text: 'm(s)' } }, height: 380, margin: { t: 40, b: 50, l: 60, r: 20 },
          }}
          style={{ width: '100%' }}
        />
      </div>
      <div className="mt-4 border-l-4 border-blue-500 pl-4 text-sm text-[var(--text-muted)]">
        <p className="font-medium text-[var(--text-strong)] mb-1">What to notice</p>
        <p>
          The L-curve bends sharply at the corner (gold diamond). Under-regularized solutions (red)
          oscillate wildly from noise amplification. Over-regularized solutions (purple) are featureless.
          The corner picks the optimal compromise. Increase noise to see the corner shift rightward.
        </p>
      </div>
      </SimulationMain>
    </SimulationPanel>
  );
}
