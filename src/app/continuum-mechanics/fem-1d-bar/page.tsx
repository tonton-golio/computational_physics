'use client';

import dynamic from 'next/dynamic';
import { useState, useEffect } from 'react';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export default function Fem1dBar() {
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
  }, [computeFEM]);

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

  const u_exact = x.map(xx => (P / (A * E)) * xx);

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
    <div style={{ padding: '20px' }}>
      <h1>Finite Element Method: 1D Axial Bar</h1>
      <p>Simulate a 1D bar fixed at left end, with axial force at right end.</p>
      <div style={{ marginBottom: '20px' }}>
        <label>Number of Elements (ne): {ne} <input type="range" min="1" max="20" value={ne} onChange={e => setNe(Number(e.target.value))} /></label><br />
        <label>Length L: {L} <input type="range" min="0.1" max="10" step="0.1" value={L} onChange={e => setL(Number(e.target.value))} /></label><br />
        <label>Cross-section Area A: {A} <input type="range" min="0.1" max="10" step="0.1" value={A} onChange={e => setA(Number(e.target.value))} /></label><br />
        <label>Young's Modulus E: {E} <input type="range" min="1" max="100" value={E} onChange={e => setE(Number(e.target.value))} /></label><br />
        <label>Force P: {P} <input type="range" min="0" max="10" step="0.1" value={P} onChange={e => setP(Number(e.target.value))} /></label><br />
        <label>Deformation Exaggeration: {exag} <input type="range" min="1" max="100" value={exag} onChange={e => setExag(Number(e.target.value))} /></label>
      </div>
      <Plot data={meshData} layout={{ title: {text: "Original and Deformed Mesh"}, xaxis: { title: {text: "Position x"} }, yaxis: { title: {text: "y (always 0)"}, range: [-0.1, 0.1] } }} />
      <Plot data={dispData} layout={{ title: {text: 'Displacement u(x)'}, xaxis: { title: {text: 'Position x'} }, yaxis: { title: {text: 'Displacement u'} } }} />
      <Plot data={strainData} layout={{ title: {text: 'Strain per Element'}, xaxis: { title: {text: 'Element'}, tickvals: strains.map((_, i) => i), ticktext: strains.map((_, i) => `E${i+1}`) }, yaxis: { title: {text: 'Strain ε'} } }} />
    </div>
  );
}