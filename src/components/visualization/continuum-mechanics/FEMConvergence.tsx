"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * FEM convergence study: error vs number of elements for the 1D bar problem.
 * Exact solution: u(x) = x - x^2/2 (unit bar, fixed at x=0, uniform load).
 * Shows how increasing element count reduces the L2 error, with theoretical
 * convergence rate O(h^(p+1)) for polynomial degree p.
 */

function gaussElim(A: number[][], b: number[]): number[] {
  const n = A.length;
  const a = A.map(row => [...row]);
  const bb = [...b];
  for (let p = 0; p < n; p++) {
    let max = p;
    for (let i = p + 1; i < n; i++) {
      if (Math.abs(a[i][p]) > Math.abs(a[max][p])) max = i;
    }
    [a[p], a[max]] = [a[max], a[p]];
    [bb[p], bb[max]] = [bb[max], bb[p]];
    for (let i = p + 1; i < n; i++) {
      const alpha = a[i][p] / a[p][p];
      bb[i] -= alpha * bb[p];
      for (let j = p; j < n; j++) a[i][j] -= alpha * a[p][j];
    }
  }
  const x = Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++) sum += a[i][j] * x[j];
    x[i] = (bb[i] - sum) / a[i][i];
  }
  return x;
}

function exactSolution(x: number) {
  return x - x * x / 2;
}

function solveFEM(ne: number): { X: number[]; U: number[] } {
  const n = ne + 1;
  const le = 1 / ne;
  const kf = 1 / le;

  const K = Array.from({ length: n }, () => Array(n).fill(0));
  for (let e = 0; e < ne; e++) {
    K[e][e] += kf;
    K[e][e + 1] -= kf;
    K[e + 1][e] -= kf;
    K[e + 1][e + 1] += kf;
  }

  const f = Array(n).fill(0);
  for (let e = 0; e < ne; e++) {
    f[e] += le / 2;
    f[e + 1] += le / 2;
  }

  // BC: u(0) = 0
  for (let i = 0; i < n; i++) { K[0][i] = 0; K[i][0] = 0; }
  K[0][0] = 1;
  f[0] = 0;

  const U = gaussElim(K, f);
  const X = Array.from({ length: n }, (_, i) => i * le);
  return { X, U };
}

function computeL2Error(ne: number): number {
  const { X, U } = solveFEM(ne);
  // L2 error: integrate (u_fem - u_exact)^2 using Simpson's rule on each element
  let errorSq = 0;
  for (let e = 0; e < ne; e++) {
    const x0 = X[e], x1 = X[e + 1];
    const u0 = U[e], u1 = U[e + 1];
    const nQuad = 10;
    const dx = (x1 - x0) / nQuad;
    for (let q = 0; q < nQuad; q++) {
      const xm = x0 + (q + 0.5) * dx;
      const t = (xm - x0) / (x1 - x0);
      const uFem = u0 * (1 - t) + u1 * t;
      const uEx = exactSolution(xm);
      errorSq += (uFem - uEx) ** 2 * dx;
    }
  }
  return Math.sqrt(errorSq);
}

export default function FEMConvergence({}: SimulationComponentProps) {
  const [maxElements, setMaxElements] = useState(32);

  const { errorTraces, rateInfo } = useMemo(() => {
    const elemCounts = [];
    for (let ne = 1; ne <= maxElements; ne++) {
      elemCounts.push(ne);
    }

    const neArr: number[] = [];
    const errArr: number[] = [];
    const hArr: number[] = [];

    for (const ne of elemCounts) {
      const err = computeL2Error(ne);
      neArr.push(ne);
      hArr.push(1 / ne);
      errArr.push(err);
    }

    // Theoretical O(h^2) reference line for linear elements
    const h2Ref: number[] = [];
    const h2X: number[] = [];
    const c = errArr[0] / (hArr[0] ** 2);
    for (const h of hArr) {
      h2X.push(1 / h); // ne
      h2Ref.push(c * h ** 2);
    }

    // Compute empirical convergence rate
    const n1 = Math.max(0, neArr.length - 3);
    const n2 = neArr.length - 1;
    let rate = 0;
    if (n2 > n1 && errArr[n1] > 0 && errArr[n2] > 0) {
      rate = Math.log(errArr[n1] / errArr[n2]) / Math.log(hArr[n1] / hArr[n2]);
    }

    const errorTraces = [
      {
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        x: neArr,
        y: errArr,
        name: 'FEM L\u00b2 error',
        line: { color: '#3b82f6', width: 2 },
        marker: { color: '#3b82f6', size: 4 },
      },
      {
        type: 'scatter' as const,
        mode: 'lines' as const,
        x: h2X,
        y: h2Ref,
        name: 'O(h\u00b2) reference',
        line: { color: '#ef4444', width: 2, dash: 'dash' as const },
      },
    ];

    return { errorTraces, rateInfo: rate };
  }, [maxElements]);

  return (
    <SimulationPanel title="FEM Convergence: Error vs Number of Elements" caption={`Linear finite elements approximate the exact solution u(x) = x \u2212 x\u00b2/2. The L\u00b2 error decreases as O(h\u00b2) for linear elements, confirming optimal convergence. Empirical rate: ${rateInfo.toFixed(2)}.`}>
      <SimulationConfig>
        <div className="max-w-xs">
          <SimulationLabel>
            Max elements: {maxElements}
          </SimulationLabel>
          <Slider min={4} max={64} step={1} value={[maxElements]}
            onValueChange={([v]) => setMaxElements(v)} className="w-full" />
        </div>
      </SimulationConfig>
      <SimulationMain>
        <CanvasChart
          data={errorTraces}
          layout={{
            xaxis: { title: { text: 'Number of elements' }, type: 'log' },
            yaxis: { title: { text: 'L\u00b2 error' }, type: 'log' },
          }}
          style={{ width: '100%', height: 400 }}
        />
      </SimulationMain>
      <p className="mt-3 text-xs text-[var(--text-muted)]">
        Blue: measured L&sup2; error. Red dashed: theoretical O(h&sup2;) convergence for
        linear (p=1) elements. On a log-log plot the slope should approach &minus;2.
        Quadratic elements (p=2) would give O(h&sup3;).
      </p>
    </SimulationPanel>
  );
}
