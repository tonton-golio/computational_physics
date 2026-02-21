'use client';

import { useState, useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { COLORS } from '@/lib/chart-colors';
import type { SimulationComponentProps } from '@/shared/types/simulation';
import type { Matrix2x2 } from './eigen-utils';

export function EigenTransformation({}: SimulationComponentProps) {
  const [matrix, setMatrix] = useState<Matrix2x2>([[2, 1], [0, 3]]);

  const { data, layout } = useMemo(() => {
    const a = matrix[0][0], b = matrix[0][1];
    const c = matrix[1][0], d = matrix[1][1];

    // Eigenvalues: lambda = (a+d +/- sqrt((a-d)^2 + 4bc)) / 2
    const trace = Math.sqrt((a - d) ** 2 + 4 * b * c);
    const lambda1 = (a + d + trace) / 2;
    const lambda2 = (a + d - trace) / 2;

    // Eigenvectors (normalized)
    let v1: [number, number], v2: [number, number];
    if (Math.abs(b) > 1e-10) {
      v1 = [1, (lambda1 - a) / b];
      v2 = [1, (lambda2 - a) / b];
    } else if (Math.abs(c) > 1e-10) {
      v1 = [(lambda1 - d) / c, 1];
      v2 = [(lambda2 - d) / c, 1];
    } else {
      v1 = [1, 0];
      v2 = [0, 1];
    }

    // Normalize
    const norm1 = Math.sqrt(v1[0] ** 2 + v1[1] ** 2);
    const norm2 = Math.sqrt(v2[0] ** 2 + v2[1] ** 2);
    v1 = [v1[0] / norm1, v1[1] / norm1];
    v2 = [v2[0] / norm2, v2[1] / norm2];

    // Scale for visualization
    const scale = 2;
    v1 = [v1[0] * scale, v1[1] * scale];
    v2 = [v2[0] * scale, v2[1] * scale];

    // Transformed eigenvectors
    const tv1: [number, number] = [a * v1[0] + b * v1[1], c * v1[0] + d * v1[1]];
    const tv2: [number, number] = [a * v2[0] + b * v2[1], c * v2[0] + d * v2[1]];

    // Unit circle points
    const nPoints = 50;
    const circleX: number[] = [];
    const circleY: number[] = [];
    const ellipseX: number[] = [];
    const ellipseY: number[] = [];

    for (let i = 0; i <= nPoints; i++) {
      const theta = (2 * Math.PI * i) / nPoints;
      const x = Math.cos(theta);
      const y = Math.sin(theta);
      circleX.push(x);
      circleY.push(y);
      ellipseX.push(a * x + b * y);
      ellipseY.push(c * x + d * y);
    }

    const data = [
      // Unit circle
      { x: circleX, y: circleY, type: 'scatter' as const, mode: 'lines' as const, line: { color: '#444', width: 1 }, name: 'Unit circle', showlegend: false },
      // Transformed ellipse
      { x: ellipseX, y: ellipseY, type: 'scatter' as const, mode: 'lines' as const, line: { color: COLORS.primary, width: 2 }, name: 'Transformed', showlegend: false },
      // Eigenvector 1 (original)
      { x: [0, v1[0]], y: [0, v1[1]], type: 'scatter' as const, mode: 'lines' as const, line: { color: COLORS.tertiary, width: 3 }, name: `v\u2081 (\u03BB\u2081=${lambda1.toFixed(2)})` },
      // Eigenvector 2 (original)
      { x: [0, v2[0]], y: [0, v2[1]], type: 'scatter' as const, mode: 'lines' as const, line: { color: COLORS.secondary, width: 3 }, name: `v\u2082 (\u03BB\u2082=${lambda2.toFixed(2)})` },
      // Transformed eigenvectors
      { x: [0, tv1[0]], y: [0, tv1[1]], type: 'scatter' as const, mode: 'lines' as const, line: { color: COLORS.tertiary, width: 2, dash: 'dot' }, showlegend: false },
      { x: [0, tv2[0]], y: [0, tv2[1]], type: 'scatter' as const, mode: 'lines' as const, line: { color: COLORS.secondary, width: 2, dash: 'dot' }, showlegend: false },
    ];

    const layout = {
      xaxis: { title: { text: 'x' }, range: [-5, 5] },
      yaxis: { title: { text: 'y' }, range: [-5, 5] },
      title: { text: 'Eigenvalue Transformation' },
    };

    return { data, layout };
  }, [matrix]);

  return (
    <div className="space-y-4">
      <CanvasChart data={data} layout={layout} style={{ width: '100%', height: 320 }} />
      <div className="flex gap-4 items-center flex-wrap">
        <span className="text-sm text-[var(--text-muted)]">Matrix A:</span>
        <input
          type="number"
          value={matrix[0][0]}
          onChange={(e) => setMatrix([[parseFloat(e.target.value) || 0, matrix[0][1]], matrix[1]])}
          className="w-16 px-2 py-1 bg-[var(--surface-1)] rounded text-[var(--text-strong)] text-center"
        />
        <input
          type="number"
          value={matrix[0][1]}
          onChange={(e) => setMatrix([[matrix[0][0], parseFloat(e.target.value) || 0], matrix[1]])}
          className="w-16 px-2 py-1 bg-[var(--surface-1)] rounded text-[var(--text-strong)] text-center"
        />
        <input
          type="number"
          value={matrix[1][0]}
          onChange={(e) => setMatrix([matrix[0], [parseFloat(e.target.value) || 0, matrix[1][1]]])}
          className="w-16 px-2 py-1 bg-[var(--surface-1)] rounded text-[var(--text-strong)] text-center"
        />
        <input
          type="number"
          value={matrix[1][1]}
          onChange={(e) => setMatrix([matrix[0], [matrix[1][0], parseFloat(e.target.value) || 0]])}
          className="w-16 px-2 py-1 bg-[var(--surface-1)] rounded text-[var(--text-strong)] text-center"
        />
        <button
          onClick={() => setMatrix([[2, 1], [0, 3]])}
          className="px-3 py-1 bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white rounded text-sm"
        >
          Reset
        </button>
      </div>
      <p className="text-xs text-[var(--text-soft)]">
        Eigenvectors (colored lines) stay on their span during transformation — they only scale by their eigenvalue.
      </p>
    </div>
  );
}

export default EigenTransformation;
