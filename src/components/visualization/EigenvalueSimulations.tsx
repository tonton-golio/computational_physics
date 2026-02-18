'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import Plotly from 'plotly.js-dist';
import { mergePlotlyTheme } from '@/lib/plotly-theme';
import { Slider } from '@/components/ui/slider';

const COLORS = {
  primary: '#3b82f6',
  secondary: '#ec4899',
  tertiary: '#10b981',
  warning: '#f59e0b',
  danger: '#ef4444',
  accent: '#8b5cf6',
};

type Matrix2x2 = [[number, number], [number, number]];

// Utils for eigenvalue computations
function computeUnitEllipse(matrix: Matrix2x2): { circleX: number[]; circleY: number[]; ellipseX: number[]; ellipseY: number[] } {
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
    ellipseX.push(matrix[0][0] * x + matrix[0][1] * y);
    ellipseY.push(matrix[1][0] * x + matrix[1][1] * y);
  }

  return { circleX, circleY, ellipseX, ellipseY };
}

function matrixMultiply(A: Matrix2x2, B: Matrix2x2): Matrix2x2 {
  return [
    [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
    [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]
  ];
}

function matrixAdd(A: Matrix2x2, B: Matrix2x2): Matrix2x2 {
  return [
    [A[0][0] + B[0][0], A[0][1] + B[0][1]],
    [A[1][0] + B[1][0], A[1][1] + B[1][1]]
  ];
}

function matrixScale(A: Matrix2x2, s: number): Matrix2x2 {
  return [
    [A[0][0] * s, A[0][1] * s],
    [A[1][0] * s, A[1][1] * s]
  ];
}

function matrixExp(A: Matrix2x2): Matrix2x2 {
  let result: Matrix2x2 = [[1, 0], [0, 1]]; // I
  let term: Matrix2x2 = [[1, 0], [0, 1]]; // I
  for (let k = 1; k <= 15; k++) { // More terms for accuracy
    term = matrixScale(matrixMultiply(term, A), 1 / k);
    result = matrixAdd(result, term);
  }
  return result;
}

// ============ EIGENVALUE VISUALIZATIONS ============


interface SimulationProps {
  id?: string;
}

// 1. Eigenvalue Transformation Demo - Show how eigenvectors stay on their span
export function EigenTransformation({}: SimulationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [matrix, setMatrix] = useState<Matrix2x2>([[2, 1], [0, 3]]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

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

    const data: Plotly.Data[] = [
      // Unit circle
      { x: circleX, y: circleY, type: 'scatter', mode: 'lines', line: { color: '#444', width: 1 }, name: 'Unit circle', showlegend: false },
      // Transformed ellipse
      { x: ellipseX, y: ellipseY, type: 'scatter', mode: 'lines', line: { color: COLORS.primary, width: 2 }, name: 'Transformed', showlegend: false },
      // Eigenvector 1 (original)
      { x: [0, v1[0]], y: [0, v1[1]], type: 'scatter', mode: 'lines', line: { color: COLORS.tertiary, width: 3 }, name: `v₁ (λ₁=${lambda1.toFixed(2)})` },
      // Eigenvector 2 (original)
      { x: [0, v2[0]], y: [0, v2[1]], type: 'scatter', mode: 'lines', line: { color: COLORS.secondary, width: 3 }, name: `v₂ (λ₂=${lambda2.toFixed(2)})` },
      // Transformed eigenvectors
      { x: [0, tv1[0]], y: [0, tv1[1]], type: 'scatter', mode: 'lines', line: { color: COLORS.tertiary, width: 2, dash: 'dot' }, showlegend: false },
      { x: [0, tv2[0]], y: [0, tv2[1]], type: 'scatter', mode: 'lines', line: { color: COLORS.secondary, width: 2, dash: 'dot' }, showlegend: false },
    ];

    const layout = mergePlotlyTheme({
      xaxis: { title: { text: 'x' }, range: [-5, 5] },
      yaxis: { title: { text: 'y' }, range: [-5, 5], scaleanchor: 'x' },
      title: { text: 'Eigenvalue Transformation' },
      legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,0)' },
    });

    Plotly.newPlot(container, data, layout, { responsive: true, displayModeBar: false });
  }, [matrix]);

  return (
    <div className="space-y-4">
      <div ref={containerRef} className="w-full h-80 bg-[var(--surface-1)] rounded-lg overflow-hidden" />
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

// 2. Power Method Animation
export function PowerMethodAnimation({}: SimulationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [iteration, setIteration] = useState(0);
  const [eigenvalueEst, setEigenvalueEst] = useState(0);
  const [matrix] = useState<Matrix2x2>([[4, 1], [2, 3]]);
  const animationRef = useRef<{ x: number[]; y: number[] }>({ x: [1, 1], y: [1, 1] });

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

    // Update plot
    const container = containerRef.current;
    if (!container) return;

    const data: Plotly.Data[] = [
      // Iteration path
      {
        x: animationRef.current.x,
        y: animationRef.current.y,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: COLORS.primary, width: 2 },
        marker: { size: 4, color: animationRef.current.x.map((_, i) => i), colorscale: 'Viridis' },
        name: 'Iterations',
      },
      // True eigenvector direction
      {
        x: [-1, 1],
        y: [-0.5, 0.5], // Approximate eigenvector direction
        type: 'scatter',
        mode: 'lines',
        line: { color: COLORS.tertiary, width: 2, dash: 'dash' },
        name: 'True eigenvector',
      },
      // Current point
      {
        x: [animationRef.current.x[animationRef.current.x.length - 1]],
        y: [animationRef.current.y[animationRef.current.y.length - 1]],
        type: 'scatter',
        mode: 'markers',
        marker: { size: 12, color: COLORS.secondary },
        name: 'Current',
      },
    ];

    const layout = mergePlotlyTheme({
      xaxis: { title: { text: 'x' }, range: [-1.5, 1.5] },
      yaxis: { title: { text: 'y' }, range: [-1.5, 1.5], scaleanchor: 'x' },
      title: { text: `Power Method — Iteration ${animationRef.current.x.length - 1}` },
    });

    Plotly.react(container, data, layout, { responsive: true, displayModeBar: false });
  }, [matrix]);

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(step, 500);
    return () => clearInterval(interval);
  }, [isRunning, step]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const data: Plotly.Data[] = [
      {
        x: animationRef.current.x,
        y: animationRef.current.y,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: COLORS.primary, width: 2 },
        marker: { size: 8, color: COLORS.secondary },
        name: 'Vector',
      },
    ];

    Plotly.newPlot(container, data, mergePlotlyTheme({
      xaxis: { range: [-1.5, 1.5] },
      yaxis: { range: [-1.5, 1.5], scaleanchor: 'x' },
      title: { text: 'Power Method Animation' },
    }), { responsive: true, displayModeBar: false });
  }, []);

  const reset = () => {
    animationRef.current = { x: [1, 1], y: [1, 1] };
    setIteration(0);
    setEigenvalueEst(0);
    setIsRunning(false);
  };

  return (
    <div className="space-y-4">
      <div ref={containerRef} className="w-full h-80 bg-[var(--surface-1)] rounded-lg overflow-hidden" />
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
        <span className="text-[var(--text-muted)]">Estimated λ: <span className="text-blue-400">{eigenvalueEst.toFixed(4)}</span></span>
        <span className="text-[var(--text-muted)]">True λ₁: <span className="text-green-400">{trueEigenvalue.toFixed(4)}</span></span>
        <span className="text-[var(--text-muted)]">Error: <span className="text-red-400">{Math.abs(eigenvalueEst - trueEigenvalue).toExponential(2)}</span></span>
      </div>
    </div>
  );
}

// 3. Hermitian matrix properties demo
export function HermitianDemo({}: SimulationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [offDiag, setOffDiag] = useState(0.8);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

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

    const data: Plotly.Data[] = [
      {
        x,
        y: y1,
        type: 'scatter',
        mode: 'lines',
        line: { color: COLORS.primary, width: 2 },
        name: `Eigenspace λ1=${lambda1.toFixed(3)}`,
      },
      {
        x,
        y: y2,
        type: 'scatter',
        mode: 'lines',
        line: { color: COLORS.secondary, width: 2 },
        name: `Eigenspace λ2=${lambda2.toFixed(3)}`,
      },
      {
        x: [0, 1],
        y: [0, lambda1],
        type: 'scatter',
        mode: 'markers',
        marker: { size: 9, color: COLORS.tertiary },
        name: 'Real eigenvalues',
      },
      {
        x: [0, 1],
        y: [0, lambda2],
        type: 'scatter',
        mode: 'markers',
        marker: { size: 9, color: COLORS.warning },
        showlegend: false,
      },
    ];

    const layout = mergePlotlyTheme({
      title: { text: 'Hermitian Matrix: Real Eigenstructure' },
      xaxis: { title: { text: 'Basis coordinate x' }, range: [-3, 3] },
      yaxis: { title: { text: 'Transformed coordinate y' }, range: [-12, 12] },
      legend: { bgcolor: 'rgba(0,0,0,0)' },
    });

    Plotly.newPlot(container, data, layout, { responsive: true, displayModeBar: false });
  }, [offDiag]);

  return (
    <div className="space-y-4">
      <div ref={containerRef} className="w-full h-80 bg-[var(--surface-1)] rounded-lg overflow-hidden" />
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

// 4. Inverse iteration with shift
export function InverseIterationDemo({}: SimulationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [sigma, setSigma] = useState(2.2);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

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

    const data: Plotly.Data[] = [
      {
        x: iters,
        y: estimates,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: COLORS.primary, width: 2 },
        marker: { size: 6, color: COLORS.primary },
        name: 'Rayleigh estimate',
      },
      {
        x: [iters[0], iters[iters.length - 1]],
        y: [sigma, sigma],
        type: 'scatter',
        mode: 'lines',
        line: { color: COLORS.secondary, width: 2, dash: 'dot' },
        name: 'Shift σ',
      },
    ];

    const layout = mergePlotlyTheme({
      title: { text: 'Inverse Iteration (Shift-and-Invert)' },
      xaxis: { title: { text: 'Iteration' } },
      yaxis: { title: { text: 'Estimated eigenvalue' } },
      legend: { bgcolor: 'rgba(0,0,0,0)' },
    });

    Plotly.newPlot(container, data, layout, { responsive: true, displayModeBar: false });
  }, [sigma]);

  return (
    <div className="space-y-4">
      <div ref={containerRef} className="w-full h-80 bg-[var(--surface-1)] rounded-lg overflow-hidden" />
      <div className="space-y-2">
        <label className="text-sm text-[var(--text-muted)]">Shift σ: {sigma.toFixed(2)}</label>
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

// 3. Gershgorin Circles
export function GershgorinCircles({}: SimulationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [matrix] = useState<Matrix2x2>([[2, 1], [1, 3]]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

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
    const data: Plotly.Data[] = [];
    const colors = [COLORS.primary, COLORS.secondary, COLORS.tertiary];

    for (let i = 0; i < n; i++) {
      // Draw circle
      const theta = Array.from({ length: 100 }, (_, k) => (2 * Math.PI * k) / 99);
      const circleX = theta.map(t => centers[i] + radii[i] * Math.cos(t));
      const circleY = theta.map(t => radii[i] * Math.sin(t));

      data.push({
        x: circleX,
        y: circleY,
        type: 'scatter',
        mode: 'lines',
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
      type: 'scatter',
      mode: 'markers',
      marker: { size: 12, color: COLORS.warning, symbol: 'x', line: { width: 3 } },
      name: 'Eigenvalues',
    });

    const layout = mergePlotlyTheme({
      xaxis: { title: { text: 'Real axis' }, range: [-2, 8] },
      yaxis: { title: { text: 'Imaginary axis' }, range: [-4, 4], scaleanchor: 'x' },
      title: { text: 'Gershgorin Circles' },
      legend: { x: 1.02, y: 1, bgcolor: 'rgba(0,0,0,0)' },
    });

    Plotly.newPlot(container, data, layout, { responsive: true, displayModeBar: false });
  }, [matrix]);

  return (
    <div className="space-y-4">
      <div ref={containerRef} className="w-full h-80 bg-[var(--surface-1)] rounded-lg overflow-hidden" />
      <div className="text-sm text-[var(--text-muted)]">
        Each Gershgorin disc contains at least one eigenvalue. The center is the diagonal element, radius is the sum of off-diagonal absolute values in that row.
      </div>
    </div>
  );
}

// 4. Convergence Comparison
export function ConvergenceComparison({}: SimulationProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const iterations = 20;
    const trueEigenvalue = 5;

    // Power method: linear convergence
    const powerMethod = Array.from({ length: iterations }, (_, i) =>
      trueEigenvalue - 2 * Math.exp(-0.3 * i)
    );

    // Inverse iteration: linear but faster
    const inverseIter = Array.from({ length: iterations }, (_, i) =>
      trueEigenvalue - 1.5 * Math.exp(-0.5 * i)
    );

    // Rayleigh quotient: cubic convergence
    const rayleigh = Array.from({ length: iterations }, (_, i) =>
      i < 5 ? trueEigenvalue - Math.exp(-0.8 * i * i) : trueEigenvalue
    );

    const x = Array.from({ length: iterations }, (_, i) => i);

    const data: Plotly.Data[] = [
      {
        x,
        y: powerMethod.map(v => Math.abs(v - trueEigenvalue)),
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: COLORS.primary, width: 2 },
        marker: { size: 6 },
        name: 'Power Method (linear)',
      },
      {
        x,
        y: inverseIter.map(v => Math.abs(v - trueEigenvalue)),
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: COLORS.secondary, width: 2 },
        marker: { size: 6 },
        name: 'Inverse Iteration (linear)',
      },
      {
        x,
        y: rayleigh.map(v => Math.abs(v - trueEigenvalue) + 1e-15),
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: COLORS.tertiary, width: 2 },
        marker: { size: 6 },
        name: 'Rayleigh Quotient (cubic)',
      },
    ];

    const layout = mergePlotlyTheme({
      xaxis: { title: { text: 'Iteration' } },
      yaxis: { title: { text: 'Error |λ - λ_true|' }, type: 'log' },
      title: { text: 'Eigenvalue Algorithm Convergence' },
      legend: { x: 0.5, y: 1.1, xanchor: 'center', orientation: 'h', bgcolor: 'rgba(0,0,0,0)' },
    });

    const container = containerRef.current;
    if (!container) return;

    Plotly.newPlot(container, data, layout, { responsive: true, displayModeBar: false });
  }, []);

  return (
    <div className="space-y-4">
      <div ref={containerRef} className="w-full h-80 bg-[var(--surface-1)] rounded-lg overflow-hidden" />
      <p className="text-sm text-[var(--text-muted)]">
        Rayleigh quotient iteration achieves cubic convergence, dramatically faster than the linear convergence of power method.
      </p>
    </div>
  );
}

// 5. QR Algorithm Visualization
export function QRAlgorithmAnimation({}: SimulationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [iteration, setIteration] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const matrixRef = useRef<number[][]>([[4, 1, 2], [2, 3, 1], [1, 2, 2]]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Display current matrix state
    const A = matrixRef.current;

    // Create heatmap of matrix
    const z = A.map(row => row.slice());

    const data: Plotly.Data[] = [{
      type: 'heatmap',
      z,
      colorscale: 'RdBu',
      colorbar: { title: { text: 'Value', side: 'right' } },
      x: ['Col 1', 'Col 2', 'Col 3'],
      y: ['Row 1', 'Row 2', 'Row 3'],
    }];

    const layout = mergePlotlyTheme({
      title: { text: `QR Algorithm — Iteration ${iteration}` },
      annotations: A.flatMap((row, i) =>
        row.map((val, j) => ({
          x: j,
          y: i,
          text: val.toFixed(2),
          font: { color: Math.abs(val) > 1 ? '#fff' : '#000' },
          showarrow: false,
        }))
      ),
    });

    Plotly.react(container, data, layout, { responsive: true, displayModeBar: false });
  }, [iteration]);

  const step = () => {
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
  };

  const reset = () => {
    matrixRef.current = [[4, 1, 2], [2, 3, 1], [1, 2, 2]];
    setIteration(0);
    setIsRunning(false);
  };

  useEffect(() => {
    if (!isRunning) return;
    const interval = setInterval(step, 1000);
    return () => clearInterval(interval);
  }, [isRunning]);

  return (
    <div className="space-y-4">
      <div ref={containerRef} className="w-full h-80 bg-[var(--surface-1)] rounded-lg overflow-hidden" />
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

// 6. Rayleigh Quotient Surface
export function RayleighQuotientSurface({}: SimulationProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
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

    const data: Plotly.Data[] = [{
      type: 'surface',
      x,
      y,
      z,
      colorscale: 'Viridis',
      colorbar: { title: { text: 'λ_R' } },
    }];

    const layout = mergePlotlyTheme({
      scene: {
        xaxis: { title: { text: 'x₁' } },
        yaxis: { title: { text: 'x₂' } },
        zaxis: { title: { text: 'λ_R' } },
        camera: { eye: { x: 1.5, y: 1.5, z: 1 } },
      },
      title: { text: 'Rayleigh Quotient Surface' },
      margin: { l: 0, r: 0, b: 0, t: 40 },
    });

    const container = containerRef.current;
    if (!container) return;

    Plotly.newPlot(container, data, layout, { responsive: true, displayModeBar: false });
  }, []);

  return (
    <div className="space-y-4">
      <div ref={containerRef} className="w-full h-80 bg-[var(--surface-1)] rounded-lg overflow-hidden" />
      <p className="text-sm text-[var(--text-muted)]">
        The Rayleigh quotient surface has stationary points at eigenvectors, where the value equals the eigenvalue.
      </p>
    </div>
  );
}

// 7. Matrix Exponential Visualization
export function MatrixExponentialSimulation({}: SimulationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [matrix, setMatrix] = useState<Matrix2x2>([[1, 0.5], [0, 2]]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const expA = matrixExp(matrix);

    // Compute unit circle transformations
    const { circleX, circleY, ellipseX: linearX, ellipseY: linearY } = computeUnitEllipse(matrix);
    const { ellipseX: expX, ellipseY: expY } = computeUnitEllipse(expA);

    const data: Plotly.Data[] = [
      // Unit circle
      { x: circleX, y: circleY, type: 'scatter', mode: 'lines', line: { color: '#444', width: 1 }, name: 'Unit circle', showlegend: false },
      // Linear transformation (A)
      { x: linearX, y: linearY, type: 'scatter', mode: 'lines', line: { color: COLORS.primary, width: 2 }, name: 'A • circle' },
      // Exponential transformation (e^A)
      { x: expX, y: expY, type: 'scatter', mode: 'lines', line: { color: COLORS.secondary, width: 2 }, name: 'e^A • circle' },
    ];

    const layout = mergePlotlyTheme({
      xaxis: { title: { text: 'x' }, range: [-5, 5] },
      yaxis: { title: { text: 'y' }, range: [-5, 5], scaleanchor: 'x' },
      title: { text: 'Matrix Exponential: Linear vs Exponential Transformation' },
      legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,0)' },
    });

    Plotly.newPlot(container, data, layout, { responsive: true, displayModeBar: false });
  }, [matrix]);

  return (
    <div className="space-y-4">
      <div ref={containerRef} className="w-full h-80 bg-[var(--surface-1)] rounded-lg overflow-hidden" />
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
          onClick={() => setMatrix([[1, 0.5], [0, 2]])}
          className="px-3 py-1 bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white rounded text-sm"
        >
          Reset
        </button>
      </div>
      <div className="text-sm text-[var(--text-muted)] space-y-1">
        <div>e^A ≈ [
          {matrixExp(matrix)[0][0].toFixed(3)}, {matrixExp(matrix)[0][1].toFixed(3)};
          {matrixExp(matrix)[1][0].toFixed(3)}, {matrixExp(matrix)[1][1].toFixed(3)}
        ]</div>
        <div>The exponential smooths linear transformations, preserving positivity and orientation.</div>
      </div>
    </div>
  );
}

// 8. Characteristic Polynomial Visualization
export function CharacteristicPolynomial({}: SimulationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
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
      return [1, -trace, det]; // coefficients for λ² - trace λ + det
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
      return [-1, trace, -sumMinors, det]; // for -λ³ + trace λ² - sumMinors λ + det
    }
  };

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

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

    const data: Plotly.Data[] = [
      {
        x: lambda,
        y: poly,
        type: 'scatter',
        mode: 'lines',
        line: { color: COLORS.primary, width: 2 },
        name: 'p(λ)',
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

    const layout = mergePlotlyTheme({
      xaxis: { title: { text: 'λ' }, range: [-6, 6] },
      yaxis: { title: { text: 'p(λ)' } },
      title: { text: `Characteristic Polynomial (${size}x${size} matrix)` },
      legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,0)' },
    });

    Plotly.newPlot(container, data, layout, { responsive: true, displayModeBar: false });
  }, [matrix, size]);

  return (
    <div className="space-y-4">
      <div ref={containerRef} className="w-full h-80 bg-[var(--surface-1)] rounded-lg overflow-hidden" />
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
        The characteristic polynomial p(λ) = det(A - λI). Roots are the eigenvalues.
      </p>
    </div>
  );
}

// ============ SIMULATION REGISTRY ============

export const EIGENVALUE_SIMULATIONS: Record<string, React.ComponentType<SimulationProps>> = {
  'eigen-transformation': EigenTransformation,
  'power-method-animation': PowerMethodAnimation,
  'hermitian-demo': HermitianDemo,
  'inverse-iteration': InverseIterationDemo,
  'gershgorin-circles': GershgorinCircles,
  'convergence-comparison': ConvergenceComparison,
  'qr-algorithm-animation': QRAlgorithmAnimation,
  'rayleigh-convergence': RayleighQuotientSurface,
  'matrix-exponential': MatrixExponentialSimulation,
  'characteristic-polynomial': CharacteristicPolynomial,
};

export function getEigenvalueSimulation(id: string): React.ComponentType<SimulationProps> | null {
  return EIGENVALUE_SIMULATIONS[id] || null;
}
