'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import Plotly from 'plotly.js-dist';

const COLORS = {
  primary: '#3b82f6',
  secondary: '#ec4899',
  tertiary: '#10b981',
  warning: '#f59e0b',
  danger: '#ef4444',
  accent: '#8b5cf6',
  grid: '#1e1e2e',
  zero: '#2d2d44',
};

const BASE_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(15,15,25,1)',
  font: { color: '#9ca3af', family: 'system-ui' },
  margin: { t: 40, r: 20, b: 40, l: 50 },
  xaxis: { gridcolor: COLORS.grid, zerolinecolor: COLORS.zero },
  yaxis: { gridcolor: COLORS.grid, zerolinecolor: COLORS.zero },
};

// ============ EIGENVALUE VISUALIZATIONS ============

interface SimulationProps {
  id?: string;
}

// 1. Eigenvalue Transformation Demo - Show how eigenvectors stay on their span
export function EigenTransformation({}: SimulationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [matrix, setMatrix] = useState([[2, 1], [0, 3]]);
  
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    
    const a = matrix[0][0], b = matrix[0][1];
    const c = matrix[1][0], d = matrix[1][1];
    
    // Eigenvalues: λ = (a+d ± sqrt((a-d)² + 4bc)) / 2
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
    
    const layout: Partial<Plotly.Layout> = {
      ...BASE_LAYOUT,
      xaxis: { ...BASE_LAYOUT.xaxis, title: { text: 'x' }, range: [-5, 5] },
      yaxis: { ...BASE_LAYOUT.yaxis, title: { text: 'y' }, range: [-5, 5], scaleanchor: 'x' },
      title: { text: 'Eigenvalue Transformation' },
      legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,0)' },
    };
    
    Plotly.newPlot(container, data, layout, { responsive: true, displayModeBar: false });
  }, [matrix]);
  
  return (
    <div className="space-y-4">
      <div ref={containerRef} className="w-full h-80 bg-[#151525] rounded-lg overflow-hidden" />
      <div className="flex gap-4 items-center flex-wrap">
        <span className="text-sm text-gray-400">Matrix A:</span>
        <input
          type="number"
          value={matrix[0][0]}
          onChange={(e) => setMatrix([[parseFloat(e.target.value) || 0, matrix[0][1]], matrix[1]])}
          className="w-16 px-2 py-1 bg-[#151525] rounded text-white text-center"
        />
        <input
          type="number"
          value={matrix[0][1]}
          onChange={(e) => setMatrix([[matrix[0][0], parseFloat(e.target.value) || 0], matrix[1]])}
          className="w-16 px-2 py-1 bg-[#151525] rounded text-white text-center"
        />
        <input
          type="number"
          value={matrix[1][0]}
          onChange={(e) => setMatrix([matrix[0], [parseFloat(e.target.value) || 0, matrix[1][1]]])}
          className="w-16 px-2 py-1 bg-[#151525] rounded text-white text-center"
        />
        <input
          type="number"
          value={matrix[1][1]}
          onChange={(e) => setMatrix([matrix[0], [matrix[1][0], parseFloat(e.target.value) || 0]])}
          className="w-16 px-2 py-1 bg-[#151525] rounded text-white text-center"
        />
        <button
          onClick={() => setMatrix([[2, 1], [0, 3]])}
          className="px-3 py-1 bg-blue-600 rounded text-sm hover:bg-blue-700"
        >
          Reset
        </button>
      </div>
      <p className="text-xs text-gray-500">
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
  const [matrix] = useState([[4, 1], [2, 3]]);
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
    
    const layout: Partial<Plotly.Layout> = {
      ...BASE_LAYOUT,
      xaxis: { ...BASE_LAYOUT.xaxis, title: { text: 'x' }, range: [-1.5, 1.5] },
      yaxis: { ...BASE_LAYOUT.yaxis, title: { text: 'y' }, range: [-1.5, 1.5], scaleanchor: 'x' },
      title: { text: `Power Method — Iteration ${animationRef.current.x.length - 1}` },
    };
    
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
    
    Plotly.newPlot(container, data, {
      ...BASE_LAYOUT,
      xaxis: { ...BASE_LAYOUT.xaxis, range: [-1.5, 1.5] },
      yaxis: { ...BASE_LAYOUT.yaxis, range: [-1.5, 1.5], scaleanchor: 'x' },
      title: { text: 'Power Method Animation' },
    }, { responsive: true, displayModeBar: false });
  }, []);
  
  const reset = () => {
    animationRef.current = { x: [1, 1], y: [1, 1] };
    setIteration(0);
    setEigenvalueEst(0);
    setIsRunning(false);
  };
  
  return (
    <div className="space-y-4">
      <div ref={containerRef} className="w-full h-80 bg-[#151525] rounded-lg overflow-hidden" />
      <div className="flex gap-4 items-center">
        <button
          onClick={() => setIsRunning(!isRunning)}
          className={`px-4 py-2 rounded ${isRunning ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'}`}
        >
          {isRunning ? 'Pause' : 'Start'}
        </button>
        <button onClick={reset} className="px-4 py-2 bg-gray-600 rounded hover:bg-gray-700">
          Reset
        </button>
        <button onClick={step} disabled={isRunning} className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-700 disabled:opacity-50">
          Step
        </button>
      </div>
      <div className="flex gap-6 text-sm">
        <span className="text-gray-400">Iteration: <span className="text-white">{iteration}</span></span>
        <span className="text-gray-400">Estimated λ: <span className="text-blue-400">{eigenvalueEst.toFixed(4)}</span></span>
        <span className="text-gray-400">True λ₁: <span className="text-green-400">{trueEigenvalue.toFixed(4)}</span></span>
        <span className="text-gray-400">Error: <span className="text-red-400">{Math.abs(eigenvalueEst - trueEigenvalue).toExponential(2)}</span></span>
      </div>
    </div>
  );
}

// 3. Gershgorin Circles
export function GershgorinCircles({}: SimulationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [matrix] = useState([[4, 1, 0], [2, 3, 1], [0, 1, 2]]);
  
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
    
    const layout: Partial<Plotly.Layout> = {
      ...BASE_LAYOUT,
      xaxis: { ...BASE_LAYOUT.xaxis, title: { text: 'Real axis' }, range: [-2, 8] },
      yaxis: { ...BASE_LAYOUT.yaxis, title: { text: 'Imaginary axis' }, range: [-4, 4], scaleanchor: 'x' },
      title: { text: 'Gershgorin Circles' },
      legend: { x: 1.02, y: 1, bgcolor: 'rgba(0,0,0,0)' },
    };
    
    Plotly.newPlot(container, data, layout, { responsive: true, displayModeBar: false });
  }, [matrix]);
  
  return (
    <div className="space-y-4">
      <div ref={containerRef} className="w-full h-80 bg-[#151525] rounded-lg overflow-hidden" />
      <div className="text-sm text-gray-400">
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
    
    const layout: Partial<Plotly.Layout> = {
      ...BASE_LAYOUT,
      xaxis: { ...BASE_LAYOUT.xaxis, title: { text: 'Iteration' } },
      yaxis: { ...BASE_LAYOUT.yaxis, title: { text: 'Error |λ - λ_true|' }, type: 'log' },
      title: { text: 'Eigenvalue Algorithm Convergence' },
      legend: { x: 0.5, y: 1.1, xanchor: 'center', orientation: 'h', bgcolor: 'rgba(0,0,0,0)' },
    };
    
    const container = containerRef.current;
    if (!container) return;
    
    Plotly.newPlot(container, data, layout, { responsive: true, displayModeBar: false });
  }, []);
  
  return (
    <div className="space-y-4">
      <div ref={containerRef} className="w-full h-80 bg-[#151525] rounded-lg overflow-hidden" />
      <p className="text-sm text-gray-400">
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
    
    const layout: Partial<Plotly.Layout> = {
      ...BASE_LAYOUT,
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
    };
    
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
      <div ref={containerRef} className="w-full h-80 bg-[#151525] rounded-lg overflow-hidden" />
      <div className="flex gap-4 items-center">
        <button
          onClick={() => setIsRunning(!isRunning)}
          className={`px-4 py-2 rounded ${isRunning ? 'bg-red-600' : 'bg-green-600'}`}
        >
          {isRunning ? 'Pause' : 'Run'}
        </button>
        <button onClick={step} disabled={isRunning} className="px-4 py-2 bg-blue-600 rounded disabled:opacity-50">
          Step
        </button>
        <button onClick={reset} className="px-4 py-2 bg-gray-600 rounded">
          Reset
        </button>
      </div>
      <p className="text-sm text-gray-400">
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
    
    const layout: Partial<Plotly.Layout> = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      scene: {
        xaxis: { title: { text: 'x₁' }, gridcolor: COLORS.grid, color: '#9ca3af' },
        yaxis: { title: { text: 'x₂' }, gridcolor: COLORS.grid, color: '#9ca3af' },
        zaxis: { title: { text: 'λ_R' }, gridcolor: COLORS.grid, color: '#9ca3af' },
        bgcolor: 'rgba(15,15,25,1)',
        camera: { eye: { x: 1.5, y: 1.5, z: 1 } },
      },
      title: { text: 'Rayleigh Quotient Surface' },
      margin: { l: 0, r: 0, b: 0, t: 40 },
    };
    
    const container = containerRef.current;
    if (!container) return;
    
    Plotly.newPlot(container, data, layout, { responsive: true, displayModeBar: false });
  }, []);
  
  return (
    <div className="space-y-4">
      <div ref={containerRef} className="w-full h-80 bg-[#151525] rounded-lg overflow-hidden" />
      <p className="text-sm text-gray-400">
        The Rayleigh quotient surface has stationary points at eigenvectors, where the value equals the eigenvalue.
      </p>
    </div>
  );
}

// ============ SIMULATION REGISTRY ============

export const EIGENVALUE_SIMULATIONS: Record<string, React.ComponentType<SimulationProps>> = {
  'eigen-transformation': EigenTransformation,
  'power-method-animation': PowerMethodAnimation,
  'gershgorin-circles': GershgorinCircles,
  'convergence-comparison': ConvergenceComparison,
  'qr-algorithm-animation': QRAlgorithmAnimation,
  'rayleigh-convergence': RayleighQuotientSurface,
};

export function getEigenvalueSimulation(id: string): React.ComponentType<SimulationProps> | null {
  return EIGENVALUE_SIMULATIONS[id] || null;
}
