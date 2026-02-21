'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { COLORS } from '@/lib/chart-colors';
import type { SimulationComponentProps } from '@/shared/types/simulation';
import type { Matrix2x2 } from './eigen-utils';

function buildPowerMethodData(xArr: number[], yArr: number[]) {
  return [
    // Iteration path
    {
      x: [...xArr],
      y: [...yArr],
      type: 'scatter' as const,
      mode: 'lines+markers' as const,
      line: { color: COLORS.primary, width: 2 },
      marker: { size: 4, color: xArr.map((_, i) => i) as number[], colorscale: 'Viridis' },
      name: 'Iterations',
    },
    // True eigenvector direction
    {
      x: [-1, 1],
      y: [-0.5, 0.5], // Approximate eigenvector direction
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: COLORS.tertiary, width: 2, dash: 'dash' },
      name: 'True eigenvector',
    },
    // Current point
    {
      x: [xArr[xArr.length - 1]],
      y: [yArr[yArr.length - 1]],
      type: 'scatter' as const,
      mode: 'markers' as const,
      marker: { size: 12, color: COLORS.secondary },
      name: 'Current',
    },
  ];
}

function buildPowerMethodLayout(iterCount: number) {
  return {
    xaxis: { title: { text: 'x' }, range: [-1.5, 1.5] },
    yaxis: { title: { text: 'y' }, range: [-1.5, 1.5] },
    title: { text: `Power Method \u2014 Iteration ${iterCount}` },
  };
}

export function PowerMethodAnimation({}: SimulationComponentProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [iteration, setIteration] = useState(0);
  const [eigenvalueEst, setEigenvalueEst] = useState(0);
  const [matrix] = useState<Matrix2x2>([[4, 1], [2, 3]]);
  const animationRef = useRef<{ x: number[]; y: number[] }>({ x: [1, 1], y: [1, 1] });
  const [chartData, setChartData] = useState<{ data: ReturnType<typeof buildPowerMethodData>; layout: ReturnType<typeof buildPowerMethodLayout> }>(() => ({
    data: buildPowerMethodData([1, 1], [1, 1]),
    layout: buildPowerMethodLayout(0),
  }));

  const trueEigenvalue = 5; // Dominant eigenvalue of [[4,1],[2,3]]

  const step = useCallback(() => {
    const a = matrix[0][0], b = matrix[0][1];
    const c = matrix[1][0], d = matrix[1][1];

    const x = animationRef.current.x;
    const y = animationRef.current.y;

    // Apply matrix
    const newX = a * x[x.length - 1] + b * y[y.length - 1];
    const newY = c * x[x.length - 1] + d * y[y.length - 1];

    // Normalize
    const norm = Math.sqrt(newX ** 2 + newY ** 2);
    animationRef.current.x.push(newX / norm);
    animationRef.current.y.push(newY / norm);

    // Rayleigh quotient
    const xK = newX / norm;
    const yK = newY / norm;
    const rq = (xK * (a * xK + b * yK) + yK * (c * xK + d * yK)) / (xK ** 2 + yK ** 2);

    setEigenvalueEst(rq);
    setIteration((i) => i + 1);

    // Update chart data via state
    const iterCount = animationRef.current.x.length - 1;
    setChartData({
      data: buildPowerMethodData(animationRef.current.x, animationRef.current.y),
      layout: buildPowerMethodLayout(iterCount),
    });
  }, [matrix]);

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(step, 500);
    return () => clearInterval(interval);
  }, [isRunning, step]);

  const reset = () => {
    animationRef.current = { x: [1, 1], y: [1, 1] };
    setIteration(0);
    setEigenvalueEst(0);
    setIsRunning(false);
    setChartData({
      data: buildPowerMethodData([1, 1], [1, 1]),
      layout: buildPowerMethodLayout(0),
    });
  };

  return (
    <div className="space-y-4">
      <CanvasChart data={chartData.data} layout={chartData.layout} style={{ width: '100%', height: 320 }} />
      <div className="flex gap-4 items-center">
        <button
          onClick={() => setIsRunning(!isRunning)}
          className={`px-4 py-2 rounded ${isRunning ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'}`}
        >
          {isRunning ? 'Pause' : 'Start'}
        </button>
        <button onClick={reset} className="px-4 py-2 bg-[var(--surface-3)] text-[var(--text-strong)] rounded hover:bg-[var(--border-strong)]">
          Reset
        </button>
        <button onClick={step} disabled={isRunning} className="px-4 py-2 bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white rounded disabled:opacity-50">
          Step
        </button>
      </div>
      <div className="flex gap-6 text-sm">
        <span className="text-[var(--text-muted)]">Iteration: <span className="text-[var(--text-strong)]">{iteration}</span></span>
        <span className="text-[var(--text-muted)]">Estimated \u03BB: <span className="text-blue-400">{eigenvalueEst.toFixed(4)}</span></span>
        <span className="text-[var(--text-muted)]">True \u03BB\u2081: <span className="text-green-400">{trueEigenvalue.toFixed(4)}</span></span>
        <span className="text-[var(--text-muted)]">Error: <span className="text-red-400">{Math.abs(eigenvalueEst - trueEigenvalue).toExponential(2)}</span></span>
      </div>
    </div>
  );
}

export default PowerMethodAnimation;
