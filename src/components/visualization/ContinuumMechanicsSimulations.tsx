'use client';

import React, { useState, useCallback, useMemo, useEffect } from 'react';
import Plotly from 'react-plotly.js';
import { Data, Layout } from 'plotly.js';

interface SimulationProps {
  id: string;
}

// Stress-Strain Simulation
function StressStrainSim({ id }: SimulationProps) {
  const [params, setParams] = useState({
    E: 200, // Young's modulus, GPa
    alpha: 0.002,
    n: 5,
    maxStrain: 0.05, // 5%
    numPoints: 100
  });

  // Bisection method to solve for sigma given epsilon
  const solveSigma = useCallback((epsilon: number, E: number, alpha: number, n: number): number => {
    if (epsilon === 0) return 0;
    let low = 0;
    let high = E * epsilon * 10; // upper bound
    let mid = 0;
    for (let i = 0; i < 100; i++) {
      mid = (low + high) / 2;
      const elastic = mid / E;
      const plastic = alpha * Math.pow(mid / E, n);
      const calcEpsilon = elastic + plastic;
      if (calcEpsilon < epsilon) {
        low = mid;
      } else {
        high = mid;
      }
    }
    return mid;
  }, []);

  const generateData = useMemo(() => {
    const { E, alpha, n, maxStrain, numPoints } = params;
    const epsilonValues: number[] = [];
    const linearSigma: number[] = [];
    const nonlinearSigma: number[] = [];

    for (let i = 0; i <= numPoints; i++) {
      const epsilon = (i / numPoints) * maxStrain;
      epsilonValues.push(epsilon);
      linearSigma.push(E * epsilon);
      nonlinearSigma.push(solveSigma(epsilon, E, alpha, n));
    }

    const data: Data[] = [
      {
        type: 'scatter',
        mode: 'lines',
        x: epsilonValues,
        y: linearSigma,
        name: 'Linear (Hooke\'s Law)',
        line: { color: '#3b82f6' }
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: epsilonValues,
        y: nonlinearSigma,
        name: 'Nonlinear (Ramberg-Osgood)',
        line: { color: '#ef4444' }
      }
    ];

    const layout: Partial<Layout> = {
      title: 'Stress-Strain Curve',
      xaxis: { title: 'Strain (ε)' },
      yaxis: { title: 'Stress (σ) [GPa]' },
      height: 500,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(15,15,25,1)',
      font: { color: '#9ca3af' }
    };

    return { data, layout };
  }, [params, solveSigma]);

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">Stress-Strain Simulation</h3>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-white">Young&apos;s Modulus (E) [GPa]: {params.E}</label>
          <input
            type="range"
            min={50}
            max={500}
            step={10}
            value={params.E}
            onChange={(e) => setParams(p => ({ ...p, E: Number(e.target.value) }))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white">Hardening Coefficient (α): {params.alpha.toFixed(3)}</label>
          <input
            type="range"
            min={0.001}
            max={0.01}
            step={0.001}
            value={params.alpha}
            onChange={(e) => setParams(p => ({ ...p, alpha: Number(e.target.value) }))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white">Hardening Exponent (n): {params.n}</label>
          <input
            type="range"
            min={1}
            max={10}
            step={0.5}
            value={params.n}
            onChange={(e) => setParams(p => ({ ...p, n: Number(e.target.value) }))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white">Max Strain: {params.maxStrain.toFixed(3)}</label>
          <input
            type="range"
            min={0.01}
            max={0.1}
            step={0.005}
            value={params.maxStrain}
            onChange={(e) => setParams(p => ({ ...p, maxStrain: Number(e.target.value) }))}
            className="w-full"
          />
        </div>
      </div>
      <Plotly
        data={generateData.data}
        layout={generateData.layout}
        config={{ displayModeBar: false }}
      />
      <div className="mt-4 text-sm text-gray-300">
        <p>The linear curve follows Hooke&apos;s law: σ = Eε</p>
        <p>The nonlinear curve uses the Ramberg-Osgood model: ε = σ/E + α(σ/E)^n</p>
      </div>
    </div>
  );
}

// FEM 1D Bar Simulation
function FEM1DBarSim({ id }: SimulationProps) {
  const [ne, setNe] = useState(5);
  const [L, setL] = useState(1);
  const [A, setA] = useState(1);
  const [E, setE] = useState(1);
  const [P, setP] = useState(1);
  const [exag, setExag] = useState(10);
  const [u, setU] = useState<number[]>([]);
  const [x, setX] = useState<number[]>([]);
  const [strains, setStrains] = useState<number[]>([]);

  useEffect(() => {
    computeFEM();
  }, [ne, L, A, E, P]);

  function computeFEM() {
    const n = ne + 1;
    const le = L / ne;
    const k_factor = A * E / le;
    const k_elem = [
      [k_factor, -k_factor],
      [-k_factor, k_factor]
    ];

    const K = Array.from({ length: n }, () => Array(n).fill(0));

    for (let e = 0; e < ne; e++) {
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
          K[e + i][e + j] += k_elem[i][j];
        }
      }
    }

    // Boundary condition: u[0] = 0
    for (let i = 0; i < n; i++) {
      K[0][i] = 0;
      K[i][0] = 0;
    }
    K[0][0] = 1;

    const f = Array(n).fill(0);
    f[n - 1] = P;

    // Solve K * u = f using Gaussian elimination
    const U = gaussElimination(K, f);
    setU(U);

    const X = [];
    for (let i = 0; i < n; i++) {
      X.push(i * le);
    }
    setX(X);

    const eps = [];
    for (let e = 0; e < ne; e++) {
      eps.push((U[e + 1] - U[e]) / le);
    }
    setStrains(eps);
  }

  function gaussElimination(A: number[][], b: number[]): number[] {
    const n = A.length;
    const a = A.map(row => [...row]);
    const bb = [...b];

    // Forward elimination
    for (let p = 0; p < n; p++) {
      // Find pivot
      let max = p;
      for (let i = p + 1; i < n; i++) {
        if (Math.abs(a[i][p]) > Math.abs(a[max][p])) max = i;
      }
      // Swap rows
      [a[p], a[max]] = [a[max], a[p]];
      [bb[p], bb[max]] = [bb[max], bb[p]];

      // Eliminate
      for (let i = p + 1; i < n; i++) {
        const alpha = a[i][p] / a[p][p];
        bb[i] -= alpha * bb[p];
        for (let j = p; j < n; j++) {
          a[i][j] -= alpha * a[j][p];
        }
      }
    }

    // Back substitution
    const x = Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      let sum = 0;
      for (let j = i + 1; j < n; j++) {
        sum += a[i][j] * x[j];
      }
      x[i] = (bb[i] - sum) / a[i][i];
    }
    return x;
  }

  const u_exact = x.map(xx => (P / (A * E)) * xx);

  const meshData = [
    {
      x: x,
      y: Array(x.length).fill(0),
      mode: 'lines+markers',
      line: { color: 'gray' },
      marker: { color: 'gray' },
      name: 'Original Mesh'
    },
    {
      x: x.map((xx, i) => xx + u[i] * exag),
      y: Array(x.length).fill(0),
      mode: 'lines+markers',
      line: { color: 'blue' },
      marker: { color: 'blue' },
      name: 'Deformed Mesh'
    }
  ];

  const dispData = [
    {
      x: x,
      y: u,
      mode: 'lines+markers',
      line: { color: 'red' },
      marker: { color: 'red' },
      name: 'FEM Displacement u(x)'
    },
    {
      x: x,
      y: u_exact,
      mode: 'lines',
      line: { color: 'black', dash: 'dash' },
      name: 'Exact Displacement u_exact(x)'
    }
  ];

  const strainData = [
    {
      x: strains.map((_, i) => i + 0.5), // Center of elements
      y: strains,
      mode: 'lines+markers',
      name: 'Strain'
    }
  ];

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">FEM 1D Bar Simulation</h3>
      <p className="text-gray-300 mb-4">Simulate a 1D bar fixed at left end, with axial force at right end.</p>
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div>
          <label className="text-white">Elements (ne): {ne}</label>
          <input
            type="range"
            min="1"
            max="20"
            value={ne}
            onChange={e => setNe(Number(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white">Length L: {L}</label>
          <input
            type="range"
            min="0.1"
            max="10"
            step="0.1"
            value={L}
            onChange={e => setL(Number(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white">Area A: {A}</label>
          <input
            type="range"
            min="0.1"
            max="10"
            step="0.1"
            value={A}
            onChange={e => setA(Number(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white">Young&apos;s E: {E}</label>
          <input
            type="range"
            min="1"
            max="100"
            value={E}
            onChange={e => setE(Number(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white">Force P: {P}</label>
          <input
            type="range"
            min="0"
            max="10"
            step="0.1"
            value={P}
            onChange={e => setP(Number(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white">Exag: {exag}</label>
          <input
            type="range"
            min="1"
            max="100"
            value={exag}
            onChange={e => setExag(Number(e.target.value))}
            className="w-full"
          />
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <Plotly
            data={meshData}
            layout={{
              title: { text: 'Deformed Shape', font: { color: '#9ca3af' } },
              xaxis: { title: { text: 'Position x', color: '#9ca3af' }, color: '#9ca3af' },
              yaxis: { title: { text: 'y', color: '#9ca3af' }, range: [-0.1, 0.1], color: '#9ca3af' },
              height: 300,
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(15,15,25,1)',
              font: { color: '#9ca3af' }
            }}
            config={{ displayModeBar: false }}
          />
        </div>
        <div>
          <Plotly
            data={dispData}
            layout={{
              title: { text: 'Displacement vs Exact', font: { color: '#9ca3af' } },
              xaxis: { title: { text: 'Position x', color: '#9ca3af' }, color: '#9ca3af' },
              yaxis: { title: { text: 'Displacement u', color: '#9ca3af' }, color: '#9ca3af' },
              height: 300,
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(15,15,25,1)',
              font: { color: '#9ca3af' }
            }}
            config={{ displayModeBar: false }}
          />
        </div>
        <div>
          <Plotly
            data={strainData}
            layout={{
              title: { text: 'Strain per Element', font: { color: '#9ca3af' } },
              xaxis: { title: { text: 'Element', color: '#9ca3af' }, tickvals: strains.map((_, i) => i), ticktext: strains.map((_, i) => `E${i+1}`), color: '#9ca3af' },
              yaxis: { title: { text: 'Strain ε', color: '#9ca3af' }, color: '#9ca3af' },
              height: 300,
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(15,15,25,1)',
              font: { color: '#9ca3af' }
            }}
            config={{ displayModeBar: false }}
          />
        </div>
      </div>
    </div>
  );
}

export const CONTINUUM_MECHANICS_SIMULATIONS: Record<string, React.ComponentType<SimulationProps>> = {
  'stress-strain-sim': StressStrainSim,
  'fem-1d-bar-sim': FEM1DBarSim,
};