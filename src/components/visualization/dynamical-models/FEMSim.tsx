'use client';

import React, { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { PlotData } from 'plotly.js';
import { Slider } from '@/components/ui/slider';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

// Dynamically import Plot to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

function solveFEM(ne: number, L: number, A: number, E: number, P: number): { nodes: number[], u: number[], deformed: number[] } {
  const n = ne + 1; // number of nodes
  const le = L / ne; // element length
  const k_factor = (A * E) / le;

  // Initialize global stiffness matrix
  const K = Array.from({ length: n }, () => Array(n).fill(0));

  // Assemble stiffness matrix
  for (let e = 0; e < ne; e++) {
    const i = e;
    const j = e + 1;
    K[i][i] += k_factor;
    K[i][j] += -k_factor;
    K[j][i] += -k_factor;
    K[j][j] += k_factor;
  }

  // Apply boundary conditions: u[0] = 0, f[n-1] = P
  const K_red = [];
  const f_red = [];
  for (let i = 1; i < n; i++) {
    K_red.push(K[i].slice(1));
    f_red.push(i === n - 1 ? P : 0);
  }

  // Simple solve for 1D (since tridiagonal, but for small n, use mathjs or simple loop)
  // For simplicity, since 1D, we can solve analytically or use gaussian elimination
  // But for code, implement a simple solver
  const u_red = solveLinearSystem(K_red, f_red);

  // Full u
  const u = [0, ...u_red];

  // Node positions
  const nodes = [];
  for (let i = 0; i < n; i++) {
    nodes.push(i * le);
  }

  // Deformed positions (exaggerate by factor, say 10 for visibility)
  const scale = 10;
  const deformed = nodes.map((x, idx) => x + scale * u[idx]);

  return { nodes, u, deformed };
}

function solveLinearSystem(A: number[][], b: number[]): number[] {
  // Simple Gaussian elimination for small systems
  const n = A.length;
  const augmented = A.map((row, i) => [...row, b[i]]);

  // Forward elimination
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const factor = augmented[j][i] / augmented[i][i];
      for (let k = i; k <= n; k++) {
        augmented[j][k] -= factor * augmented[i][k];
      }
    }
  }

  // Back substitution
  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = augmented[i][n];
    for (let j = i + 1; j < n; j++) {
      x[i] -= augmented[i][j] * x[j];
    }
    x[i] /= augmented[i][i];
  }

  return x;
}

export function FEMSim() {
  const [ne, setNe] = useState([10]);
  const [L, setL] = useState([1.0]);
  const [A, setA] = useState([0.01]);
  const [E, setE] = useState([200e9]);
  const [P, setP] = useState([1000]);

  const meshData = useMemo<PlotData[]>(() => {
    const { nodes, deformed } = solveFEM(ne[0], L[0], A[0], E[0], P[0]);
    return [
      {
        x: nodes,
        y: nodes.map(() => 0),
        mode: 'lines+markers',
        name: 'Original Mesh',
        line: { color: 'gray' },
        marker: { color: 'gray' },
      },
      {
        x: deformed,
        y: deformed.map(() => 0),
        mode: 'lines+markers',
        name: 'Deformed Mesh',
        line: { color: 'blue' },
        marker: { color: 'blue' },
      },
    ];
  }, [ne, L, A, E, P]);

  const displacementData = useMemo<PlotData[]>(() => {
    const { nodes, u } = solveFEM(ne[0], L[0], A[0], E[0], P[0]);
    return [
      {
        x: nodes,
        y: u,
        mode: 'lines+markers',
        name: 'Displacement',
        line: { color: 'red' },
        marker: { color: 'red' },
      },
    ];
  }, [ne, L, A, E, P]);

  return (
    <div className="space-y-8">
      <Card>
        <CardHeader>
          <CardTitle>FEM 1D Bar Simulation</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div>
              <h3 className="text-lg font-semibold mb-4">Mesh (Original and Deformed)</h3>
              {meshData && (
                <Plot
                  data={meshData}
                  layout={{
                    width: 500,
                    height: 300,
                    xaxis: { title: 'Position x' },
                    yaxis: { title: 'Deformation (exaggerated)' },
                    margin: { t: 20, b: 40, l: 60, r: 20 },
                  }}
                  config={{ displayModeBar: false }}
                />
              )}
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-4">Displacement u(x)</h3>
              {displacementData && (
                <Plot
                  data={displacementData}
                  layout={{
                    width: 500,
                    height: 300,
                    xaxis: { title: 'Position x' },
                    yaxis: { title: 'Displacement u' },
                    margin: { t: 20, b: 40, l: 60, r: 20 },
                  }}
                  config={{ displayModeBar: false }}
                />
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Parameters</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <label className="text-sm font-medium">Number of Elements (ne): {ne[0]}</label>
            <Slider value={ne} onValueChange={setNe} min={1} max={50} step={1} />
          </div>
          <div>
            <label className="text-sm font-medium">Length L: {L[0].toFixed(2)} m</label>
            <Slider value={L} onValueChange={setL} min={0.1} max={10} step={0.1} />
          </div>
          <div>
            <label className="text-sm font-medium">Area A: {A[0].toExponential(2)} m²</label>
            <Slider value={A} onValueChange={setA} min={1e-6} max={1e-2} step={1e-6} />
          </div>
          <div>
            <label className="text-sm font-medium">Young&apos;s Modulus E: {E[0].toExponential(0)} Pa</label>
            <Slider value={E} onValueChange={setE} min={1e9} max={1e12} step={1e9} />
          </div>
          <div>
            <label className="text-sm font-medium">Force P: {P[0].toFixed(0)} N</label>
            <Slider value={P} onValueChange={setP} min={0} max={10000} step={10} />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}