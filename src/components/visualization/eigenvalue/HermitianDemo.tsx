'use client';

import { useState, useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';
import { COLORS } from '@/lib/chart-colors';
import type { SimulationComponentProps } from '@/shared/types/simulation';
import type { Matrix2x2 } from './eigen-utils';

export function HermitianDemo({}: SimulationComponentProps) {
  const [offDiag, setOffDiag] = useState(0.8);

  const { data, layout } = useMemo(() => {
    // Real symmetric matrix is the Hermitian subset over R.
    const A: Matrix2x2 = [[2.5, offDiag], [offDiag, 4.0]];
    const a = A[0][0];
    const b = A[0][1];
    const d = A[1][1];
    const disc = Math.sqrt((a - d) ** 2 + 4 * b * b);
    const lambda1 = (a + d + disc) / 2;
    const lambda2 = (a + d - disc) / 2;

    const x = Array.from({ length: 200 }, (_, i) => -5 + (10 * i) / 199);
    const y1 = x.map((v) => lambda1 * v);
    const y2 = x.map((v) => lambda2 * v);

    const data = [
      {
        x,
        y: y1,
        type: 'scatter' as const,
        mode: 'lines' as const,
        line: { color: COLORS.primary, width: 2 },
        name: `Eigenspace \u03BB1=${lambda1.toFixed(3)}`,
      },
      {
        x,
        y: y2,
        type: 'scatter' as const,
        mode: 'lines' as const,
        line: { color: COLORS.secondary, width: 2 },
        name: `Eigenspace \u03BB2=${lambda2.toFixed(3)}`,
      },
      {
        x: [0, 1],
        y: [0, lambda1],
        type: 'scatter' as const,
        mode: 'markers' as const,
        marker: { size: 9, color: COLORS.tertiary },
        name: 'Real eigenvalues',
      },
      {
        x: [0, 1],
        y: [0, lambda2],
        type: 'scatter' as const,
        mode: 'markers' as const,
        marker: { size: 9, color: COLORS.warning },
        showlegend: false,
      },
    ];

    const layout = {
      title: { text: 'Hermitian Matrix: Real Eigenstructure' },
      xaxis: { title: { text: 'Basis coordinate x' }, range: [-3, 3] },
      yaxis: { title: { text: 'Transformed coordinate y' }, range: [-12, 12] },
    };

    return { data, layout };
  }, [offDiag]);

  return (
    <div className="space-y-4">
      <CanvasChart data={data} layout={layout} style={{ width: '100%', height: 320 }} />
      <div className="space-y-2">
        <label className="text-sm text-[var(--text-muted)]">Symmetric off-diagonal coupling: {offDiag.toFixed(2)}</label>
        <Slider
          min={-2}
          max={2}
          step={0.05}
          value={[offDiag]}
          onValueChange={([v]) => setOffDiag(v)}
          className="w-full"
        />
      </div>
      <p className="text-xs text-[var(--text-soft)]">
        Hermitian matrices have real eigenvalues and orthogonal eigenvectors, yielding numerically stable eigendecompositions.
      </p>
    </div>
  );
}

export default HermitianDemo;
