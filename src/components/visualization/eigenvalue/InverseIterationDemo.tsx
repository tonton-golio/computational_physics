'use client';

import { useState, useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';
import { COLORS } from '@/lib/chart-colors';
import type { SimulationComponentProps } from '@/shared/types/simulation';

export function InverseIterationDemo({}: SimulationComponentProps) {
  const [sigma, setSigma] = useState(2.2);

  const { data, layout } = useMemo(() => {
    const A: number[][] = [
      [4, 1, 0],
      [1, 3, 1],
      [0, 1, 2],
    ];

    const dot = (u: number[], v: number[]) => u.reduce((s, ui, i) => s + ui * v[i], 0);
    const norm = (v: number[]) => Math.sqrt(dot(v, v));

    const solve3 = (M: number[][], b: number[]): number[] => {
      const a = M.map((row) => row.slice());
      const rhs = b.slice();
      const n = 3;
      for (let p = 0; p < n; p++) {
        let pivot = p;
        for (let i = p + 1; i < n; i++) {
          if (Math.abs(a[i][p]) > Math.abs(a[pivot][p])) pivot = i;
        }
        [a[p], a[pivot]] = [a[pivot], a[p]];
        [rhs[p], rhs[pivot]] = [rhs[pivot], rhs[p]];
        const denom = a[p][p] || 1e-12;
        for (let i = p + 1; i < n; i++) {
          const m = a[i][p] / denom;
          for (let j = p; j < n; j++) a[i][j] -= m * a[p][j];
          rhs[i] -= m * rhs[p];
        }
      }
      const x = [0, 0, 0];
      for (let i = n - 1; i >= 0; i--) {
        let s = rhs[i];
        for (let j = i + 1; j < n; j++) s -= a[i][j] * x[j];
        x[i] = s / (a[i][i] || 1e-12);
      }
      return x;
    };

    let x = [1, 0.2, -0.4];
    const estimates: number[] = [];
    const iters: number[] = [];

    for (let k = 0; k < 14; k++) {
      const M = [
        [A[0][0] - sigma, A[0][1], A[0][2]],
        [A[1][0], A[1][1] - sigma, A[1][2]],
        [A[2][0], A[2][1], A[2][2] - sigma],
      ];
      const y = solve3(M, x);
      const yn = norm(y) || 1;
      x = y.map((v) => v / yn);
      const Ax = [
        A[0][0] * x[0] + A[0][1] * x[1] + A[0][2] * x[2],
        A[1][0] * x[0] + A[1][1] * x[1] + A[1][2] * x[2],
        A[2][0] * x[0] + A[2][1] * x[1] + A[2][2] * x[2],
      ];
      const rq = dot(x, Ax);
      estimates.push(rq);
      iters.push(k + 1);
    }

    const data = [
      {
        x: iters,
        y: estimates,
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        line: { color: COLORS.primary, width: 2 },
        marker: { size: 6, color: COLORS.primary },
        name: 'Rayleigh estimate',
      },
      {
        x: [iters[0], iters[iters.length - 1]],
        y: [sigma, sigma],
        type: 'scatter' as const,
        mode: 'lines' as const,
        line: { color: COLORS.secondary, width: 2, dash: 'dot' },
        name: 'Shift \u03C3',
      },
    ];

    const layout = {
      title: { text: 'Inverse Iteration (Shift-and-Invert)' },
      xaxis: { title: { text: 'Iteration' } },
      yaxis: { title: { text: 'Estimated eigenvalue' } },
    };

    return { data, layout };
  }, [sigma]);

  return (
    <div className="space-y-4">
      <CanvasChart data={data} layout={layout} style={{ width: '100%', height: 320 }} />
      <div className="space-y-2">
        <label className="text-sm text-[var(--text-muted)]">Shift \u03C3: {sigma.toFixed(2)}</label>
        <Slider
          min={0.5}
          max={4.5}
          step={0.05}
          value={[sigma]}
          onValueChange={([v]) => setSigma(v)}
          className="w-full"
        />
      </div>
      <p className="text-xs text-[var(--text-soft)]">
        The iteration converges to the eigenvalue nearest the chosen shift, making interior eigenvalues tractable.
      </p>
    </div>
  );
}

export default InverseIterationDemo;
