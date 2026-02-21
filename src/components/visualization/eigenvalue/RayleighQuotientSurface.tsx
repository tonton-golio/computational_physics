'use client';

import { useMemo } from 'react';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import type { SimulationComponentProps } from '@/shared/types/simulation';

export function RayleighQuotientSurface({}: SimulationComponentProps) {
  const { data, layout } = useMemo(() => {
    // Matrix: [[4, 1], [2, 3]]
    const a = 4, b = 1, c = 2, d = 3;

    const n = 50;
    const x = Array.from({ length: n }, (_, i) => -2 + (4 * i) / (n - 1));
    const y = x.slice();

    const z: number[][] = [];
    for (let i = 0; i < n; i++) {
      z[i] = [];
      for (let j = 0; j < n; j++) {
        const xi = x[j], yi = y[i];
        const denom = xi ** 2 + yi ** 2;
        if (denom < 0.01) {
          z[i][j] = 0;
        } else {
          // Rayleigh quotient: (x^T A x) / (x^T x)
          const Ax = [a * xi + b * yi, c * xi + d * yi];
          z[i][j] = (xi * Ax[0] + yi * Ax[1]) / denom;
        }
      }
    }

    const data = [{
      z,
      x,
      y,
      colorscale: 'Viridis',
      showscale: true,
    }];

    const layout = {
      title: { text: 'Rayleigh Quotient Surface' },
      xaxis: { title: { text: 'x\u2081' } },
      yaxis: { title: { text: 'x\u2082' } },
      margin: { l: 60, r: 50, b: 50, t: 40 },
    };

    return { data, layout };
  }, []);

  return (
    <div className="space-y-4">
      <CanvasHeatmap data={data} layout={layout} style={{ width: '100%', height: 320 }} />
      <p className="text-sm text-[var(--text-muted)]">
        The Rayleigh quotient surface has stationary points at eigenvectors, where the value equals the eigenvalue.
      </p>
    </div>
  );
}

export default RayleighQuotientSurface;
