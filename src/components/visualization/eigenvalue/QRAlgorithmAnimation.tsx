'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import type { SimulationComponentProps } from '@/shared/types/simulation';

function buildQRChartState(A: number[][], iter: number) {
  const z = A.map(row => row.slice());

  const data = [{
    z,
    colorscale: 'RdBu',
    showscale: true,
    x: ['Col 1', 'Col 2', 'Col 3'],
    y: ['Row 1', 'Row 2', 'Row 3'],
  }];

  const layout = {
    title: { text: `QR Algorithm \u2014 Iteration ${iter}` },
    annotations: A.flatMap((row, i) =>
      row.map((val, j) => ({
        x: j,
        y: i,
        text: val.toFixed(2),
        font: { color: Math.abs(val) > 1 ? '#fff' : '#000' },
        showarrow: false,
      }))
    ),
  };

  return { data, layout };
}

export function QRAlgorithmAnimation({}: SimulationComponentProps) {
  const [iteration, setIteration] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const matrixRef = useRef<number[][]>([[4, 1, 2], [2, 3, 1], [1, 2, 2]]);
  const [chartState, setChartState] = useState<{ data: Array<{ z: number[][]; x: string[]; y: string[]; colorscale: string; showscale: boolean }>; layout: { title: { text: string }; annotations: Array<{ x: number; y: number; text: string; font: { color: string }; showarrow: boolean }> } }>(() => buildQRChartState(matrixRef.current, 0));

  const step = useCallback(() => {
    // One QR iteration
    const A = matrixRef.current;
    const n = A.length;

    // Simple QR factorization (Gram-Schmidt for demo)
    // This is a simplified version - real QR uses Householder
    const Q = Array.from({ length: n }, () => Array(n).fill(0));
    const R = Array.from({ length: n }, () => Array(n).fill(0));

    // Copy A to R
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        R[i][j] = A[i][j];
      }
    }

    // Gram-Schmidt
    for (let j = 0; j < n; j++) {
      // Get column j
      const col = A.map(row => row[j]);

      // Subtract projections onto previous Q columns
      const qCol = [...col];
      for (let i = 0; i < j; i++) {
        let dot = 0;
        for (let k = 0; k < n; k++) {
          dot += Q[k][i] * col[k];
        }
        R[i][j] = dot;
        for (let k = 0; k < n; k++) {
          qCol[k] -= dot * Q[k][i];
        }
      }

      // Normalize
      let norm = 0;
      for (let k = 0; k < n; k++) {
        norm += qCol[k] ** 2;
      }
      norm = Math.sqrt(norm);
      R[j][j] = norm;

      for (let k = 0; k < n; k++) {
        Q[k][j] = qCol[k] / (norm || 1);
      }
    }

    // Compute RQ (next iterate)
    const newA = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        for (let k = 0; k < n; k++) {
          newA[i][j] += R[i][k] * Q[k][j];
        }
      }
    }

    matrixRef.current = newA;
    setIteration(i => i + 1);
    setChartState(buildQRChartState(newA, iteration + 1));
  }, [iteration]);

  const reset = () => {
    matrixRef.current = [[4, 1, 2], [2, 3, 1], [1, 2, 2]];
    setIteration(0);
    setIsRunning(false);
    setChartState(buildQRChartState([[4, 1, 2], [2, 3, 1], [1, 2, 2]], 0));
  };

  useEffect(() => {
    if (!isRunning) return;
    const interval = setInterval(step, 1000);
    return () => clearInterval(interval);
  }, [isRunning, step]);

  return (
    <div className="space-y-4">
      <CanvasHeatmap data={chartState.data} layout={chartState.layout} style={{ width: '100%', height: 320 }} />
      <div className="flex gap-4 items-center">
        <button
          onClick={() => setIsRunning(!isRunning)}
          className={`px-4 py-2 rounded ${isRunning ? 'bg-red-600' : 'bg-green-600'}`}
        >
          {isRunning ? 'Pause' : 'Run'}
        </button>
        <button onClick={step} disabled={isRunning} className="px-4 py-2 bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white rounded disabled:opacity-50">
          Step
        </button>
        <button onClick={reset} className="px-4 py-2 bg-[var(--surface-3)] text-[var(--text-strong)] rounded hover:bg-[var(--border-strong)]">
          Reset
        </button>
      </div>
      <p className="text-sm text-[var(--text-muted)]">
        QR iteration: A = QR, then A&apos; = RQ. The matrix converges to upper triangular with eigenvalues on the diagonal.
      </p>
    </div>
  );
}

export default QRAlgorithmAnimation;
