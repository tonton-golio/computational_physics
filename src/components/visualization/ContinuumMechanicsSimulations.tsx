'use client';

import React, { useState, useCallback, useMemo, useEffect } from 'react';
import Plotly from 'react-plotly.js';
import { Data, Layout } from 'plotly.js';
import { usePlotlyTheme } from '@/lib/plotly-theme';
import { Slider } from '@/components/ui/slider';
import StressStrainCurve from './continuum-mechanics/StressStrainCurve';
import StressTensor from './continuum-mechanics/StressTensor';
import ElasticWave from './continuum-mechanics/ElasticWave';

interface SimulationProps {
  id: string;
}

// Stress-Strain Simulation (legacy, kept for backward compatibility)
function StressStrainSim({ }: SimulationProps) {
  const [params, setParams] = useState({
    E: 200, // Young's modulus, GPa
    alpha: 0.002,
    n: 5,
    maxStrain: 0.05, // 5%
    numPoints: 100
  });
  const { mergeLayout } = usePlotlyTheme();

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

    const layout: Partial<Layout> = mergeLayout({
      title: { text: 'Stress-Strain Curve' },
      xaxis: { title: { text: 'Strain (\u03b5)' } },
      yaxis: { title: { text: 'Stress (\u03c3) [GPa]' } },
      height: 500,
    });

    return { data, layout };
  }, [params, solveSigma, mergeLayout]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Stress-Strain Simulation</h3>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-[var(--text-strong)]">Young&apos;s Modulus (E) [GPa]: {params.E}</label>
          <Slider
            min={50}
            max={500}
            step={10}
            value={[params.E]}
            onValueChange={([v]) => setParams(p => ({ ...p, E: v }))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-[var(--text-strong)]">Hardening Coefficient (&alpha;): {params.alpha.toFixed(3)}</label>
          <Slider
            min={0.001}
            max={0.01}
            step={0.001}
            value={[params.alpha]}
            onValueChange={([v]) => setParams(p => ({ ...p, alpha: v }))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-[var(--text-strong)]">Hardening Exponent (n): {params.n}</label>
          <Slider
            min={1}
            max={10}
            step={0.5}
            value={[params.n]}
            onValueChange={([v]) => setParams(p => ({ ...p, n: v }))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-[var(--text-strong)]">Max Strain: {params.maxStrain.toFixed(3)}</label>
          <Slider
            min={0.01}
            max={0.1}
            step={0.005}
            value={[params.maxStrain]}
            onValueChange={([v]) => setParams(p => ({ ...p, maxStrain: v }))}
            className="w-full"
          />
        </div>
      </div>
      <Plotly
        data={generateData.data}
        layout={generateData.layout}
        config={{ displayModeBar: false }}
      />
      <div className="mt-4 text-sm text-[var(--text-muted)]">
        <p>The linear curve follows Hooke&apos;s law: &sigma; = E&epsilon;</p>
        <p>The nonlinear curve uses the Ramberg-Osgood model: &epsilon; = &sigma;/E + &alpha;(&sigma;/E)^n</p>
      </div>
    </div>
  );
}

// FEM 1D Bar Simulation
function FEM1DBarSim({ }: SimulationProps) {
  const [ne, setNe] = useState(5);
  const [L, setL] = useState(1);
  const [A, setA] = useState(1);
  const [E, setE] = useState(1);
  const [P, setP] = useState(1);
  const [exag, setExag] = useState(10);
  const [u, setU] = useState<number[]>([]);
  const [x, setX] = useState<number[]>([]);
  const [strains, setStrains] = useState<number[]>([]);
  const { mergeLayout } = usePlotlyTheme();

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

  const computeFEM = useCallback(() => {
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
  }, [ne, L, A, E, P]);

  useEffect(() => {
    computeFEM();
  }, [ne, L, A, E, P, computeFEM]);

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
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">FEM 1D Bar Simulation</h3>
      <p className="text-[var(--text-muted)] mb-4">Simulate a 1D bar fixed at left end, with axial force at right end.</p>
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div>
          <label className="text-[var(--text-strong)]">Elements (ne): {ne}</label>
          <Slider
            min={1}
            max={20}
            value={[ne]}
            onValueChange={([v]) => setNe(v)}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-[var(--text-strong)]">Length L: {L}</label>
          <Slider
            min={0.1}
            max={10}
            step={0.1}
            value={[L]}
            onValueChange={([v]) => setL(v)}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-[var(--text-strong)]">Area A: {A}</label>
          <Slider
            min={0.1}
            max={10}
            step={0.1}
            value={[A]}
            onValueChange={([v]) => setA(v)}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-[var(--text-strong)]">Young&apos;s E: {E}</label>
          <Slider
            min={1}
            max={100}
            value={[E]}
            onValueChange={([v]) => setE(v)}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-[var(--text-strong)]">Force P: {P}</label>
          <Slider
            min={0}
            max={10}
            step={0.1}
            value={[P]}
            onValueChange={([v]) => setP(v)}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-[var(--text-strong)]">Exag: {exag}</label>
          <Slider
            min={1}
            max={100}
            value={[exag]}
            onValueChange={([v]) => setExag(v)}
            className="w-full"
          />
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <Plotly
            data={meshData}
            layout={mergeLayout({
              title: { text: 'Deformed Shape' },
              xaxis: { title: { text: 'Position x' } },
              yaxis: { title: { text: 'y' }, range: [-0.1, 0.1] },
              height: 300,
            })}
            config={{ displayModeBar: false }}
          />
        </div>
        <div>
          <Plotly
            data={dispData}
            layout={mergeLayout({
              title: { text: 'Displacement vs Exact' },
              xaxis: { title: { text: 'Position x' } },
              yaxis: { title: { text: 'Displacement u' } },
              height: 300,
            })}
            config={{ displayModeBar: false }}
          />
        </div>
        <div>
          <Plotly
            data={strainData}
            layout={mergeLayout({
              title: { text: 'Strain per Element' },
              xaxis: { title: { text: 'Element' }, tickvals: strains.map((_, i) => i), ticktext: strains.map((_, i) => `E${i+1}`) },
              yaxis: { title: { text: 'Strain \u03b5' } },
              height: 300,
            })}
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
  'stress-strain-curve': StressStrainCurve,
  'mohr-circle': StressTensor,
  'elastic-wave': ElasticWave,
};
