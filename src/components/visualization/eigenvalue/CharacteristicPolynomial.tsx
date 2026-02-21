'use client';

import { useEffect, useState, useMemo } from 'react';
import { CanvasChart, type ChartTrace } from '@/components/ui/canvas-chart';
import { COLORS } from '@/lib/chart-colors';
import type { SimulationComponentProps } from '@/shared/types/simulation';

export function CharacteristicPolynomial({}: SimulationComponentProps) {
  const [size, setSize] = useState(2);
  const [matrix, setMatrix] = useState<number[][]>([[2, 1], [0, 3]]);

  useEffect(() => {
    if (size === 2) {
      setMatrix([[2, 1], [0, 3]]);
    } else {
      setMatrix([[2, 1, 0], [0, 3, 1], [1, 0, 2]]);
    }
  }, [size]);

  const computeCharPoly = (A: number[][]) => {
    if (A.length === 2) {
      const a = A[0][0], b = A[0][1], c = A[1][0], d = A[1][1];
      const trace = a + d;
      const det = a * d - b * c;
      return [1, -trace, det]; // coefficients for \u03BB\u00B2 - trace \u03BB + det
    } else {
      const a = A[0][0], b = A[0][1], c = A[0][2];
      const d = A[1][0], e = A[1][1], f = A[1][2];
      const g = A[2][0], h = A[2][1], i = A[2][2];
      const trace = a + e + i;
      const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
      const m11 = e * i - f * h;
      const m22 = a * i - c * g;
      const m33 = a * e - b * d;
      const sumMinors = m11 + m22 + m33;
      return [-1, trace, -sumMinors, det]; // for -\u03BB\u00B3 + trace \u03BB\u00B2 - sumMinors \u03BB + det
    }
  };

  const { data, layout } = useMemo(() => {
    const coeffs = computeCharPoly(matrix);
    const lambda = Array.from({ length: 200 }, (_, i) => -6 + (12 * i) / 199);
    const poly = lambda.map(l => {
      let val = 0;
      for (let k = 0; k < coeffs.length; k++) {
        val += coeffs[k] * Math.pow(l, coeffs.length - 1 - k);
      }
      return val;
    });

    let eigenvalues: number[] = [];
    if (matrix.length === 2) {
      const a = matrix[0][0], b = matrix[0][1], c = matrix[1][0], d = matrix[1][1];
      const disc = (a + d) ** 2 - 4 * (a * d - b * c);
      if (disc >= 0) {
        eigenvalues = [(a + d + Math.sqrt(disc)) / 2, (a + d - Math.sqrt(disc)) / 2];
      }
    }

    const data: ChartTrace[] = [
      {
        x: lambda,
        y: poly,
        type: 'scatter',
        mode: 'lines',
        line: { color: COLORS.primary, width: 2 },
        name: 'p(\u03BB)',
      },
    ];
    if (eigenvalues.length) {
      data.push({
        x: eigenvalues,
        y: eigenvalues.map(() => 0),
        type: 'scatter',
        mode: 'markers',
        marker: { size: 10, color: COLORS.secondary, symbol: 'x', line: { width: 2 } },
        name: 'Roots (eigenvalues)',
      });
    }

    const layout = {
      xaxis: { title: { text: '\u03BB' }, range: [-6, 6] },
      yaxis: { title: { text: 'p(\u03BB)' } },
      title: { text: `Characteristic Polynomial (${size}x${size} matrix)` },
    };

    return { data, layout };
  }, [matrix, size]);

  return (
    <div className="space-y-4">
      <CanvasChart data={data} layout={layout} style={{ width: '100%', height: 320 }} />
      <div className="flex gap-4 items-center flex-wrap">
        <span className="text-sm text-[var(--text-muted)]">Matrix size:</span>
        <select
          value={size}
          onChange={(e) => setSize(parseInt(e.target.value))}
          className="bg-[var(--surface-1)] text-[var(--text-strong)] rounded px-2 py-1"
        >
          <option value={2}>2x2</option>
          <option value={3}>3x3</option>
        </select>
      </div>
      <div className={`grid gap-2 ${size === 2 ? 'grid-cols-2 max-w-xs' : 'grid-cols-3 max-w-md'}`}>
        {matrix.map((row, i) =>
          row.map((val, j) => (
            <input
              key={`${i}-${j}`}
              type="number"
              step="0.1"
              value={val}
              onChange={(e) => {
                const newMatrix = matrix.map(r => [...r]);
                newMatrix[i][j] = parseFloat(e.target.value) || 0;
                setMatrix(newMatrix);
              }}
              className="w-16 px-2 py-1 bg-[var(--surface-1)] rounded text-[var(--text-strong)] text-center"
            />
          ))
        )}
      </div>
      <div className="flex gap-2">
        <button
          onClick={() => {
            if (size === 2) setMatrix([[1, 0], [0, 1]]);
            else setMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
          }}
          className="px-3 py-1 bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white rounded text-sm"
        >
          Identity
        </button>
        <button
          onClick={() => {
            if (size === 2) setMatrix([[0, 0], [0, 0]]);
            else setMatrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]]);
          }}
          className="px-3 py-1 bg-[var(--surface-3)] text-[var(--text-strong)] rounded text-sm hover:bg-[var(--border-strong)]"
        >
          Zero
        </button>
      </div>
      <p className="text-xs text-[var(--text-soft)]">
        The characteristic polynomial p(\u03BB) = det(A - \u03BBI). Roots are the eigenvalues.
      </p>
    </div>
  );
}

export default CharacteristicPolynomial;
