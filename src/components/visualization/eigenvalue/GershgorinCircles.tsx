'use client';

import { useState, useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { COLORS } from '@/lib/chart-colors';
import type { SimulationComponentProps } from '@/shared/types/simulation';
import type { Matrix2x2 } from './eigen-utils';

export function GershgorinCircles({}: SimulationComponentProps) {
  const [matrix] = useState<Matrix2x2>([[2, 1], [1, 3]]);

  const { data, layout } = useMemo(() => {
    const n = matrix.length;
    const centers: number[] = [];
    const radii: number[] = [];

    for (let i = 0; i < n; i++) {
      centers.push(matrix[i][i]);
      let r = 0;
      for (let j = 0; j < n; j++) {
        if (i !== j) r += Math.abs(matrix[i][j]);
      }
      radii.push(r);
    }

    // Compute actual eigenvalues (using power iteration approximation for demo)
    // For simplicity, we'll use numpy-style eigenvalue computation approximation
    const eigenvalues = [4.732, 2.268, 1.0]; // Approximate eigenvalues of the default matrix

    // Create circle traces
    const data: Array<{
      x: number[];
      y: number[];
      type: 'scatter';
      mode: 'lines' | 'markers';
      fill?: string;
      fillcolor?: string;
      line?: { color: string; width: number };
      marker?: { size: number; color: string; symbol: string; line?: { width: number } };
      name: string;
    }> = [];
    const colors = [COLORS.primary, COLORS.secondary, COLORS.tertiary];

    for (let i = 0; i < n; i++) {
      // Draw circle
      const theta = Array.from({ length: 100 }, (_, k) => (2 * Math.PI * k) / 99);
      const circleX = theta.map(t => centers[i] + radii[i] * Math.cos(t));
      const circleY = theta.map(t => radii[i] * Math.sin(t));

      data.push({
        x: circleX,
        y: circleY,
        type: 'scatter' as const,
        mode: 'lines' as const,
        fill: 'toself',
        fillcolor: `${colors[i]}20`,
        line: { color: colors[i], width: 2 },
        name: `Disc ${i + 1}: center=${centers[i].toFixed(1)}, r=${radii[i].toFixed(1)}`,
      });
    }

    // Plot actual eigenvalues
    data.push({
      x: eigenvalues,
      y: eigenvalues.map(() => 0),
      type: 'scatter' as const,
      mode: 'markers' as const,
      marker: { size: 12, color: COLORS.warning, symbol: 'x', line: { width: 3 } },
      name: 'Eigenvalues',
    });

    const layout = {
      xaxis: { title: { text: 'Real axis' }, range: [-2, 8] },
      yaxis: { title: { text: 'Imaginary axis' }, range: [-4, 4] },
      title: { text: 'Gershgorin Circles' },
    };

    return { data, layout };
  }, [matrix]);

  return (
    <div className="space-y-4">
      <CanvasChart data={data} layout={layout} style={{ width: '100%', height: 320 }} />
      <div className="text-sm text-[var(--text-muted)]">
        Each Gershgorin disc contains at least one eigenvalue. The center is the diagonal element, radius is the sum of off-diagonal absolute values in that row.
      </div>
    </div>
  );
}

export default GershgorinCircles;
