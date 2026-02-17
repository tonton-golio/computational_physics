'use client';

import { useState, useMemo, useRef, useEffect } from 'react';
import Plotly from 'plotly.js-dist';
import type { PlotData, Layout } from 'plotly.js';

const COLORS = {
  original: '#9ca3af', // gray
  deformed: '#3b82f6', // blue
  displacement: '#10b981', // green
};

const BASE_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(15,15,25,1)',
  font: { color: '#9ca3af', family: 'system-ui' },
  margin: { t: 40, r: 20, b: 40, l: 50 },
  xaxis: {
    gridcolor: '#1e1e2e',
    zerolinecolor: '#2d2d44',
  },
  yaxis: {
    gridcolor: '#1e1e2e',
    zerolinecolor: '#2d2d44',
  },
};

export function FEM1D() {
  const [L, setL] = useState(1); // Length
  const [ne, setNe] = useState(5); // Number of elements
  const [E, setE] = useState(50); // Young's modulus
  const [A, setA] = useState(1); // Cross-section area
  const [P, setP] = useState(5); // Force at right end
  const exag = 10; // Exaggeration factor

  const femData = useMemo(() => {
    const n = ne + 1; // Number of nodes
    const h = L / ne; // Element length

    // Node coordinates
    const nodes = Array.from({ length: n }, (_, i) => i * h);

    // Element stiffness matrix
    const k_e = (A * E) / h;
    const K_e = [
      [k_e, -k_e],
      [-k_e, k_e]
    ];

    // Global stiffness matrix
    const K = Array.from({ length: n }, () => Array(n).fill(0));
    for (let e = 0; e < ne; e++) {
      const i = e;
      const j = e + 1;
      K[i][i] += K_e[0][0];
      K[i][j] += K_e[0][1];
      K[j][i] += K_e[1][0];
      K[j][j] += K_e[1][1];
    }

    // Force vector
    const F = Array(n).fill(0);
    F[n - 1] = P; // Force at right end

    // Boundary conditions: u[0] = 0
    const K_red = [];
    const F_red = [];
    for (let i = 1; i < n; i++) {
      K_red.push([]);
      F_red.push(F[i]);
      for (let j = 1; j < n; j++) {
        K_red[i - 1].push(K[i][j]);
      }
    }

    // Solve Ku = F using simple Gaussian elimination for small matrix
    const u_red = solveLinearSystem(K_red, F_red);

    // Displacements
    const u_disp = Array(n).fill(0);
    u_disp[0] = 0;
    for (let i = 1; i < n; i++) {
      u_disp[i] = u_red[i - 1];
    }

    // Deformed nodes
    const deformed = nodes.map((x, i) => x + exag * u_disp[i]);

    return { nodes, deformed, u_disp };
  }, [L, ne, E, A, P]);

  const plotData = useMemo(() => {
    const { nodes, deformed, u_disp } = femData;

    // Original mesh
    const originalData = {
      x: nodes,
      y: Array(nodes.length).fill(0),
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: COLORS.original, width: 2 },
      name: 'Original mesh'
    };

    // Deformed mesh
    const deformedData = {
      x: deformed,
      y: Array(deformed.length).fill(0),
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: COLORS.deformed, width: 3 },
      name: 'Deformed mesh (exaggerated)'
    };

    // Displacement plot
    const dispData = {
      x: nodes,
      y: u_disp,
      type: 'scatter' as const,
      mode: 'lines+markers' as const,
      line: { color: COLORS.displacement, width: 2 },
      marker: { color: COLORS.displacement, size: 6 },
      name: 'Displacement u(x)',
      xaxis: 'x2',
      yaxis: 'y2'
    };

    return [originalData, deformedData, dispData];
  }, [femData]);

  const layout = useMemo(() => ({
    ...BASE_LAYOUT,
    title: { text: '1D FEM Bar Element Analysis' },
    xaxis: { ...BASE_LAYOUT.xaxis, title: { text: 'Position x' }, domain: [0, 1] },
    yaxis: { title: { text: 'Deformation' }, domain: [0.5, 1] },
    xaxis2: { title: { text: 'Position x' }, domain: [0, 1], anchor: 'y2' },
    yaxis2: { title: { text: 'Displacement u(x)' }, domain: [0, 0.5] },
    grid: { rows: 2, columns: 1, pattern: 'independent' },
  }), []);

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <div className="flex gap-4 items-center">
          <label className="flex flex-col">
            ne: {ne}
            <input type="range" min="1" max="10" value={ne} onChange={(e) => setNe(parseInt(e.target.value))} className="w-32" />
          </label>
          <label className="flex flex-col">
            L: {L.toFixed(1)}
            <input type="range" min="0.1" max="2" step="0.1" value={L} onChange={(e) => setL(parseFloat(e.target.value))} className="w-32" />
          </label>
          <label className="flex flex-col">
            A: {A}
            <input type="range" min="1" max="10" value={A} onChange={(e) => setA(parseFloat(e.target.value))} className="w-32" />
          </label>
          <label className="flex flex-col">
            E: {E}
            <input type="range" min="1" max="100" value={E} onChange={(e) => setE(parseFloat(e.target.value))} className="w-32" />
          </label>
          <label className="flex flex-col">
            P: {P}
            <input type="range" min="1" max="10" value={P} onChange={(e) => setP(parseFloat(e.target.value))} className="w-32" />
          </label>
        </div>
      </div>
      <div className="w-full h-96 bg-black rounded-lg">
        <PlotlyPlot data={plotData} layout={layout} />
      </div>
      <p className="text-sm text-gray-400">
        1D FEM analysis of axial bar. Fixed at left (u=0), axial force at right. Shows original mesh (gray), deformed mesh (blue, exaggerated), and displacement u(x) plot.
      </p>
    </div>
  );
}

// Simple Gaussian elimination for small matrices
function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  const augmented = A.map((row, i) => [...row, b[i]]);

  // Forward elimination
  for (let i = 0; i < n; i++) {
    // Find pivot
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
        maxRow = k;
      }
    }
    // Swap
    [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];

    // Eliminate
    for (let k = i + 1; k < n; k++) {
      const factor = augmented[k][i] / augmented[i][i];
      for (let j = i; j < n + 1; j++) {
        augmented[k][j] -= factor * augmented[i][j];
      }
    }
  }

  // Back substitution
  const x = Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = augmented[i][n] / augmented[i][i];
    for (let k = i - 1; k >= 0; k--) {
      augmented[k][n] -= augmented[k][i] * x[i];
    }
  }

  return x;
}

// Helper component for Plotly
function PlotlyPlot({ data, layout }: { data: PlotData[]; layout: Partial<Layout> }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    Plotly.newPlot(container, data, layout, {
      responsive: true,
      displayModeBar: false,
    });

    return () => {
      Plotly.purge(container);
    };
  }, [data, layout]);

  return <div ref={containerRef} className="w-full h-full" />;
}